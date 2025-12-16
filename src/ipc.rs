use shared_memory::*;
use std::sync::Mutex;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QueryPacket {
    pub flag: u8, // 0 = Empty, 1 = Ready, 2 = Reading
    pub codes: [i32; 8],
    pub tri_pos: [f32; 512],
    pub mass: f32, // Added for ghost mass control
    pub text_len: u32,
    pub text: [u8; 256],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ResponsePacket {
    pub flag: u8, // 0 = Empty, 1 = Ready
    pub num_results: u32,
    pub token_ids: [u32; 10000],  // Expanded to 10000 for full output (Rainbow Tune)
    pub scores: [f32; 10000],     // Expanded to 10000
}

pub struct IpcListener {
    shmem: Shmem,
}

#[derive(Debug)]
pub enum IpcMessage {
    Query {
        text: String,
        codes: Vec<i32>,
        pos: [f32; 512],
        mass: f32,
    },
    InjectParticle {
        text: String,
        codes: Vec<i32>,
        pos: [f32; 512],
        mass: f32,
    },
    GenerativeMode {
        seed_text: String,
        codes: Vec<i32>,
        seed_pos: [f32; 512],
        mass: f32,
    },
    SaveBonds(String),
    SetGoal {
        text: String,
        vector: [f32; 512],
    },
    EnableMemory(bool),
}

impl IpcListener {
    pub fn new(suffix: Option<&str>) -> Result<Self, ShmemError> {
        let name = match suffix {
            Some(s) => format!("physics_lang_ipc_{}", s),
            None => "physics_lang_ipc".to_string(),
        };
        eprintln!("debug: Opening SHM: {}", name);

        let shmem = match ShmemConf::new().size(262144).os_id(&name).open() {
            Ok(m) => m,
            Err(_) => {
                // If not found, create it (owner)
                ShmemConf::new()
                    .size(262144)  // Expanded to 256KB for 10k results
                    .os_id(&name)
                    .create()?
            }
        };

        // On startup, aggressively clear both the query and response flags so that
        // stale values from crashed or previous runs do not leave the client
        // thinking the engine is permanently busy.
        unsafe {
            let base_ptr = shmem.as_ptr() as *mut u8;

            // Zero QueryPacket.flag
            let query_ptr = base_ptr as *mut QueryPacket;
            let query_packet = &mut *query_ptr;
            std::ptr::write_volatile(&mut query_packet.flag, 0u8);

            // Zero ResponsePacket.flag at RESPONSE_OFFSET = 65536
            let resp_ptr = base_ptr.add(65536) as *mut ResponsePacket;
            let resp_packet = &mut *resp_ptr;
            std::ptr::write_volatile(&mut resp_packet.flag, 0u8);
        }

        Ok(Self { shmem })
    }

    pub fn poll(&self) -> Option<IpcMessage> {
        unsafe {
            let ptr = self.shmem.as_ptr() as *mut QueryPacket;
            let packet = &mut *ptr;

            let flag = std::ptr::read_volatile(&packet.flag);
            
            if flag != 0 {
                eprintln!("IPC Poll: Found flag {}", flag);
            }

            if flag >= 1 && flag <= 7 {
                // Read data
                // Mark as reading (we use 3 to avoid confusion with flag 2)
                std::ptr::write_volatile(&mut packet.flag, 3); 

                let len = packet.text_len as usize;
                let text_bytes = &packet.text[..len.min(256)];
                let text = String::from_utf8_lossy(text_bytes).to_string();

                let codes = packet.codes.to_vec();
                let tri_pos = packet.tri_pos;
                let mass = packet.mass;

                std::ptr::write_volatile(&mut packet.flag, 0); // Reset to empty

                if flag == 1 {
                    return Some(IpcMessage::Query { text, codes, pos: tri_pos, mass });
                } else if flag == 2 {
                    return Some(IpcMessage::InjectParticle { text, codes, pos: tri_pos, mass });
                } else if flag == 4 {
                    return Some(IpcMessage::GenerativeMode { 
                        seed_text: text,
                        codes,
                        seed_pos: tri_pos,
                        mass 
                    });
                } else if flag == 5 {
                    return Some(IpcMessage::SaveBonds(text));
                } else if flag == 6 {
                    return Some(IpcMessage::SetGoal { text, vector: tri_pos });
                } else if flag == 7 {
                    let enable = codes[0] != 0;
                    return Some(IpcMessage::EnableMemory(enable));
                }
            }
        }
        None
    }

    pub fn write_response(&self, results: &[(usize, f32)]) {
        unsafe {
            let ptr = self.shmem.as_ptr() as *mut u8;
            // Offset 65536 for ResponsePacket
            let response_ptr = ptr.add(65536) as *mut ResponsePacket;
            let packet = &mut *response_ptr;

            packet.num_results = results.len().min(10000) as u32;

            for (i, (id, score)) in results.iter().take(10000).enumerate() {
                packet.token_ids[i] = *id as u32;
                packet.scores[i] = *score;
            }

            std::ptr::write_volatile(&mut packet.flag, 1); // Ready
        }
    }
}
