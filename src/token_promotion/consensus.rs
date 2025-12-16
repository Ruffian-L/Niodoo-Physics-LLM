//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use ed25519_dalek::{Signature, Signer, SigningKey};
use rand::random;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use super::TokenCandidate;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub String);

#[derive(Debug, Clone)]
pub struct ConsensusVote {
    pub approved: bool,
    pub votes_for: usize,
    pub votes_against: usize,
    pub node_signatures: Vec<(NodeId, Signature)>,
    pub merged_operations: usize,
}

#[derive(Clone, Debug)]
struct VocabularyOperation {
    op_id: OperationId,
    token_hash: [u8; 32],
    author: NodeId,
    clock: VersionVector,
    signature: Signature,
    score: f64,
}

#[derive(Clone, Debug, Default)]
struct VocabularyCrdt {
    operations: BTreeMap<OperationId, VocabularyOperation>,
}

impl VocabularyCrdt {
    fn insert(&mut self, op: VocabularyOperation) -> bool {
        let op_id = op.op_id;
        if self.operations.contains_key(&op_id) {
            return false;
        }
        self.operations.insert(op_id, op);
        true
    }
}

#[derive(Clone, Debug, Default)]
struct VersionVector(BTreeMap<String, u64>);

impl VersionVector {
    fn increment(&mut self, node: &NodeId) -> u64 {
        let counter = self.0.entry(node.0.clone()).or_insert(0);
        *counter += 1;
        *counter
    }

    fn merge(&mut self, other: &VersionVector) {
        for (node, value) in &other.0 {
            let entry = self.0.entry(node.clone()).or_insert(0);
            if value > entry {
                *entry = *value;
            }
        }
    }

    fn merged_with(&self, node: &NodeId, counter: u64) -> VersionVector {
        let mut next = self.clone();
        next.0.insert(node.0.clone(), counter);
        next
    }
}

type OperationId = [u8; 32];

#[derive(Default)]
struct CrdtState {
    clock: VersionVector,
    crdt: VocabularyCrdt,
}

impl CrdtState {
    fn new_operation(
        &mut self,
        token_hash: [u8; 32],
        score: f64,
        signing_key: &SigningKey,
        node_id: &NodeId,
    ) -> VocabularyOperation {
        let counter = self.clock.increment(node_id);
        let clock_snapshot = self.clock.merged_with(node_id, counter);

        let mut payload = Vec::with_capacity(token_hash.len() + node_id.0.len() + 8);
        payload.extend_from_slice(&token_hash);
        payload.extend_from_slice(node_id.0.as_bytes());
        payload.extend_from_slice(&counter.to_le_bytes());

        let signature = signing_key.sign(&payload);
        let op_id = Sha256::digest(&payload).into();

        VocabularyOperation {
            op_id,
            token_hash,
            author: node_id.clone(),
            clock: clock_snapshot,
            signature,
            score,
        }
    }

    fn apply_operation(&mut self, op: VocabularyOperation) -> bool {
        let merged = self.crdt.insert(op.clone());
        if merged {
            self.clock.merge(&op.clock);
        }
        merged
    }

    fn merge_operation(&mut self, op: VocabularyOperation) -> bool {
        self.apply_operation(op)
    }
}

pub struct ConsensusEngine {
    node_id: NodeId,
    signing_key: SigningKey,
    score_threshold: f64,
    state: Mutex<CrdtState>,
    peers: Vec<SimulatedPeer>,
}

impl ConsensusEngine {
    pub fn new(node_id: NodeId, score_threshold: f64) -> Self {
        let secret: [u8; 32] = random();
        let signing_key = SigningKey::from_bytes(&secret);
        let peers = vec![
            SimulatedPeer::new("node_1", -0.05),
            SimulatedPeer::new("node_2", 0.05),
        ];

        Self {
            node_id,
            signing_key,
            score_threshold,
            state: Mutex::new(CrdtState::default()),
            peers,
        }
    }

    pub async fn propose_token(&self, candidate: &TokenCandidate) -> Result<ConsensusVote> {
        let token_hash = Self::hash_candidate(candidate);
        let score = candidate.promotion_score();
        let total_nodes = self.peers.len() + 1;
        let mut approvals = 0usize;
        let mut signatures = Vec::new();
        let mut merged_operations = 0usize;
        let mut collected_ops = Vec::new();

        let local_operation = if score >= self.score_threshold {
            let mut state = self.state.lock().await;
            let op = state.new_operation(token_hash, score, &self.signing_key, &self.node_id);
            if state.apply_operation(op.clone()) {
                merged_operations += 1;
            }
            approvals += 1;
            signatures.push((self.node_id.clone(), op.signature.clone()));
            collected_ops.push(op.clone());
            Some(op)
        } else {
            None
        };

        let mut peer_results = HashMap::new();
        for peer in &self.peers {
            let decision = peer
                .handle_candidate(
                    token_hash,
                    score,
                    self.score_threshold,
                    local_operation.as_ref(),
                )
                .await;
            peer_results.insert(peer.id.clone(), decision);
        }

        for (peer_id, decision) in peer_results {
            match decision {
                PeerDecision::Approved(op) => {
                    approvals += 1;
                    signatures.push((peer_id.clone(), op.signature.clone()));
                    if self.merge_operation(op.clone()).await {
                        merged_operations += 1;
                    }
                    collected_ops.push(op);
                }
                PeerDecision::Rejected => {}
            }
        }

        for peer in &self.peers {
            peer.merge_external_ops(&collected_ops).await;
        }

        let votes_for = approvals;
        let votes_against = total_nodes.saturating_sub(votes_for);
        let approval_ratio = if total_nodes == 0 {
            0.0
        } else {
            votes_for as f64 / total_nodes as f64
        };
        let approved = approval_ratio >= self.score_threshold.min(1.0);

        Ok(ConsensusVote {
            approved,
            votes_for,
            votes_against,
            node_signatures: signatures,
            merged_operations,
        })
    }

    async fn merge_operation(&self, op: VocabularyOperation) -> bool {
        let mut state = self.state.lock().await;
        state.merge_operation(op)
    }

    fn hash_candidate(candidate: &TokenCandidate) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&candidate.bytes);
        hasher.update(&candidate.persistence.to_le_bytes());
        hasher.update(&candidate.frequency.to_le_bytes());
        hasher.update(&candidate.emotional_coherence.to_le_bytes());
        hasher.update(&candidate.spatial_locality.to_le_bytes());
        hasher.finalize().into()
    }
}

enum PeerDecision {
    Approved(VocabularyOperation),
    Rejected,
}

struct SimulatedPeer {
    id: NodeId,
    signing_key: SigningKey,
    bias: f64,
    state: Mutex<CrdtState>,
}

impl SimulatedPeer {
    fn new(id: impl Into<String>, bias: f64) -> Self {
        let secret: [u8; 32] = random();
        let signing_key = SigningKey::from_bytes(&secret);
        Self {
            id: NodeId(id.into()),
            signing_key,
            bias,
            state: Mutex::new(CrdtState::default()),
        }
    }

    async fn handle_candidate(
        &self,
        token_hash: [u8; 32],
        score: f64,
        threshold: f64,
        incoming: Option<&VocabularyOperation>,
    ) -> PeerDecision {
        let mut state = self.state.lock().await;
        if let Some(op) = incoming {
            state.merge_operation(op.clone());
        }

        if score + self.bias >= threshold {
            let op = state.new_operation(token_hash, score, &self.signing_key, &self.id);
            let _ = state.apply_operation(op.clone());
            PeerDecision::Approved(op)
        } else {
            PeerDecision::Rejected
        }
    }

    async fn merge_external_ops(&self, ops: &[VocabularyOperation]) {
        let mut state = self.state.lock().await;
        for op in ops {
            let _ = state.merge_operation(op.clone());
        }
    }
}
