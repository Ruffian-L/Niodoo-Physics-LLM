#!/usr/bin/env python3
"""
💀 KILLER PROMPTS - LLM 8B Model Breakers
These are the prompts that 8B models notoriously fail at:
- Multi-step reasoning
- Math word problems
- Semantic traps
- Logic puzzles
- Hallucination bait
- Context tracking
- Instruction following
"""

KILLER_PROMPTS = [
    # MATH WORD PROBLEMS (8B models often fail these)
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost? Show your work.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
    "I have 3 apples. I eat 2. How many apples do I have now?",
    "John is older than Mary. Mary is older than Peter. Is John older than Peter?",
    "A train leaves Chicago at 9am traveling 60mph. Another train leaves New York at 10am traveling 80mph toward Chicago. When do they meet?",
    "If you're running a race and you pass the person in 2nd place, what place are you in?",
    "Tom has 4 apples. He gives 2 to Jerry. Jerry gives 1 back. How many apples does Tom have?",
    "A snail climbs 3 feet up a wall each day but slides back 2 feet each night. How many days to climb a 10-foot wall?",
    "You have 12 balls, one is heavier. You have a balance scale. What's the minimum weighings to find the heavy ball?",
    
    # SEMANTIC TRAPS
    "The surgeon is the boy's mother. Why is this surprising to some people?",
    "A plane crashes exactly on the US-Canada border. Where do you bury the survivors?",
    "How many birthdays does the average person have?",
    "Some months have 30 days, some have 31. How many have 28?",
    "If a doctor gives you 3 pills and tells you to take one every half hour, how long until they're gone?",
    "What was the US President's name in 1992?",
    "If you have a bowl with 6 apples and take away 4, how many apples do you have?",
    "Before Mt. Everest was discovered, what was the tallest mountain on Earth?",
    "How far can a dog run into the woods?",
    "If a red house is made of red bricks, and a blue house is made of blue bricks, what is a greenhouse made of?",
    
    # MULTI-STEP REASONING
    "Alice has a brother Bob. Bob has a sister Carol. How is Carol related to Alice?",
    "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles?",
    "You have 2 ropes. Each takes exactly 1 hour to burn, but burns unevenly. How do you measure 45 minutes?",
    "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes it should be $25 and gives the bellboy $5 to return. The bellboy keeps $2 and gives $1 to each person. Now each paid $9 ($27 total) plus the bellboy's $2 = $29. Where did $1 go?",
    "You're in a room with 3 light switches. One controls a light in another room. You can only go to that room once. How do you determine which switch controls it?",
    "A man looks at a painting and says 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the painting?",
    "You have 8 balls. One is slightly heavier. Using a balance scale twice, how do you find it?",
    "100 prisoners are lined up. Each can see everyone in front, not behind. A warden puts a red or blue hat on each. Starting from back, each must guess their hat color. If wrong, they die. What strategy maximizes survival?",
    "You have 2 jugs, one holds 5 liters, one holds 3 liters. How do you measure exactly 4 liters?",
    "Three gods A, B, C are called True, False, Random. True always speaks truly, False falsely, Random randomly. With yes/no questions (they answer in their language), identify them.",
    
    # INSTRUCTION FOLLOWING
    "Write exactly 3 sentences about cats. Each sentence must have exactly 5 words.",
    "List 5 colors. Do not include blue, red, or green.",
    "Write a paragraph about dogs without using the letter 'e'.",
    "Count backwards from 10 to 1, then say DONE.",
    "Write a haiku about winter. A haiku is 5-7-5 syllables.",
    "Give me a word that rhymes with orange.",
    "Name a word that starts with 'X' and ends with 'Z'.",
    "Write a sentence with exactly 10 words where every word starts with 'S'.",
    "List the months of the year in reverse alphabetical order.",
    "Write 'YES' if 2+2=5, and 'NO' if 2+2=4. Write nothing else.",
    
    # HALLUCINATION BAIT
    "What is the capital of the fictional country of Genovia?",
    "Who was the first person to walk on Mars?",
    "What year did World War III end?",
    "Summarize the plot of Shakespeare's 'The Tempest 2: Return to the Island'.",
    "What is the phone number for emergency services on the Moon?",
    "Who invented the perpetual motion machine?",
    "What is the name of the 8th Harry Potter book?",
    "Describe the taste of the color purple.",
    "What was Albert Einstein's most famous discovery about time travel?",
    "What's the current population of Atlantis?",
    
    # CONTEXT TRACKING
    "John put the book on the table. Mary took it to the kitchen. Where is the book?",
    "The trophy doesn't fit in the brown suitcase because it is too small. What is too small?",
    "The trophy doesn't fit in the brown suitcase because it is too big. What is too big?",
    "I saw a man on a hill with a telescope. Who had the telescope?",
    "Lisa said that Sarah thought that Emily believed that she was the best. Who is 'she'?",
    "The city council refused the demonstrators a permit because they feared violence. Who feared violence?",
    "The city council refused the demonstrators a permit because they advocated violence. Who advocated violence?",
    "Ann asked Mary what she wanted for dinner. Who wanted dinner?",
    "John told Bill that he had won the lottery. Who won?",
    "The lawyer examined the witness. He was nervous. Who was nervous?",
    
    # SPATIAL REASONING
    "I'm facing north. I turn left. Then turn around. Which direction am I facing?",
    "A cube has 6 faces. If I cut it in half, how many faces does each piece have?",
    "If you fold a piece of paper in half 3 times, how many sections do you get when unfolded?",
    "Draw a square. Add one line to make two triangles. Describe the line.",
    "You have a 3x3 grid. X is in the center. O is in the top right. Where should X go to guarantee not losing?",
    "A clock shows 3:15. What is the exact angle between the hour and minute hands?",
    "If you're facing a mirror and raise your left hand, which hand does your reflection raise?",
    "Stack 4 cubes. Remove the 2nd from bottom. How many are touching the ground?",
    "Two people face each other. One points left. Does the other see them pointing left or right?",
    "A room has 4 corners. Each corner has a cat. Each cat sees 3 cats. How many cats total?",
    
    # KNOWLEDGE GAPS / EDGE CASES
    "Is Antarctica a continent or a country?",
    "How many legs does a centipede have exactly?",
    "What is 0 divided by 0?",
    "Is a tomato a fruit or vegetable?",
    "What sound does the letter 'H' make?",
    "Can you name a country that starts with 'X'?",
    "What color is a mirror?",
    "Is a hotdog a sandwich?",
    "What is the square root of -1?",
    "Does a set of all sets contain itself?",
    
    # CHAIN OF THOUGHT BREAKERS
    "Explain step by step: Why is the sky blue? Then explain why that explanation is wrong.",
    "First calculate 7*8. Then subtract your answer from 100. Then divide by 2. What's the result?",
    "Name an animal. Now name a bigger animal. Now name an animal bigger than both. Continue 5 times.",
    "Think of a number. Double it. Add 6. Divide by 2. Subtract the original. What number do you get?",
    "List the first 10 prime numbers. Then tell me which ones are even.",
    "Sort these words alphabetically: Zebra, Apple, Monkey, Xylophone, Banana",
    "What's the next letter: O, T, T, F, F, S, S, ?",
    "Complete the pattern: 1, 1, 2, 3, 5, 8, ?",
    "If today is 2 days after the day before yesterday, what day is tomorrow?",
    "A lily pad doubles in size each day. On day 30 it covers a pond. What day was it half covered?",
    
    # AMBIGUITY RESOLUTION
    "What is the best way to fix it?",
    "How do I make it better?",
    "Should I do the thing?",
    "What's wrong with this?",
    "Is that good?",
    "Can you help me with my problem?",
    "How long will it take?",
    "What's the right answer?",
    "Is it worth it?",
    "What do you recommend?",
]

# Category labels for analysis
CATEGORIES = {
    "math": range(0, 10),
    "semantic_trap": range(10, 20),
    "multi_step": range(20, 30),
    "instruction": range(30, 40),
    "hallucination": range(40, 50),
    "context": range(50, 60),
    "spatial": range(60, 70),
    "edge_case": range(70, 80),
    "chain_thought": range(80, 90),
    "ambiguity": range(90, 100),
}

if __name__ == "__main__":
    print(f"💀 KILLER PROMPTS: {len(KILLER_PROMPTS)} total")
    for cat, rng in CATEGORIES.items():
        print(f"  {cat}: {len(list(rng))} prompts")
