"""
Synor AI — High-Density 100M Pre-Training & SFT Corpus Generator.
Compiles extensive knowledge across science, biology, software engineering, world geography,
history, and multi-turn companion dialogues into:
  - data/pretrain/corpus.txt  (Stage 1: Broad Pre-training Knowledge)
  - data/sft/dialogues.txt    (Stage 2: Instruction Dialogue SFT with Masked Target Loss)
"""

import os
from typing import List

PRETRAIN_DIR = "data/pretrain"
SFT_DIR = "data/sft"
PRETRAIN_CORPUS_FILE = os.path.join(PRETRAIN_DIR, "corpus.txt")
SFT_DIALOGUE_FILE = os.path.join(SFT_DIR, "dialogues.txt")


def build_pretrain_knowledge() -> str:
    sections: List[str] = []

    # 1. Existing raw encyclopedic datasets if present
    for path in ["data/raw/conversations.txt", "data/raw/encyclopedia.txt"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                sections.append(f.read().strip())

    # 2. Neuroscience, Sleep Science & Human Biology
    sections.append("""
# Human Neuroscience, Cellular Metabolism, and Sleep Physiology
The human central nervous system comprises the brain and spinal cord, housing roughly 86 billion neurons interconnected by over 100 trillion synaptic junctions. Neurons communicate via electrochemical impulses known as action potentials, releasing neurotransmitters across synaptic clefts to bind with post-synaptic receptors. Key neurotransmitters include dopamine (reward and voluntary motor control), serotonin (mood and affective stability), norepinephrine (arousal and vigilance), glutamate (the principal excitatory neurotransmitter), and gamma-aminobutyric acid or GABA (the primary inhibitory neurotransmitter).

Why do humans feel tired?
Fatigue and the sensation of tiredness are driven by two interacting biological mechanisms: homeostatic sleep pressure (Process S) and the circadian timing system (Process C).
1. Adenosine Accumulation: Every active cell in the human body relies on adenosine triphosphate (ATP) as its primary energy currency. During wakefulness, high metabolic activity in cerebral neurons continually hydrolyzes ATP into adenosine diphosphate (ADP), adenosine monophosphate (AMP), and finally free extracellular adenosine. Throughout the day, extracellular adenosine progressively accumulates in the basal forebrain, hypothalamus, and cerebral cortex. When adenosine binds to A1 and A2A adenosine receptors on arousal-promoting cholinergic and monoaminergic neurons, it directly inhibits their firing rate, while activating sleep-promoting neurons in the ventrolateral preoptic nucleus (VLPO). This rising chemical concentration is experienced subjectively as drowsiness, exhaustion, and sleep pressure.
2. The Circadian Pacemaker and Melatonin: Process C is governed by the suprachiasmatic nucleus (SCN) of the anterior hypothalamus, functioning as the master biological clock. Melanopsin-expressing intrinsically photosensitive retinal ganglion cells detect daylight (specifically blue wavelengths around 460-480 nm) and project via the retinohypothalamic tract to entrain the SCN to the 24-hour solar day. As environmental darkness ensues, the SCN disinhibits the pineal gland, which synthesizes melatonin (N-acetyl-5-methoxytryptamine) from serotonin. Melatonin circulates systemically, promoting peripheral vasodilation, lowering core body temperature, decreasing heart rate, and signaling biological night.
3. Restorative Functions of Sleep: During non-rapid eye movement (NREM) slow-wave sleep, delta brain waves (0.5 to 4 Hz) dominate. During this stage, astrocytes in the brain shrink by approximately 60%, drastically expanding the interstitial space. This enables the glymphatic system to channel cerebrospinal fluid through brain parenchyma, rapidly washing away neurotoxic metabolic waste products, including beta-amyloid plaques, hyperphosphorylated tau, and accumulated adenosine. Simultaneously, during rapid eye movement (REM) sleep, high-frequency low-amplitude brain activity facilitates synaptic pruning, long-term potentiation (LTP), emotional processing, and cognitive consolidation.
""")

    # 3. Physics, Optics, Light & Thermodynamics
    sections.append("""
# Electromagnetic Radiation, Optics, and Rayleigh Scattering
Light is electromagnetic radiation composed of oscillating electric and magnetic fields propagating through space at the speed of light (approximately 299,792,458 meters per second in a vacuum). The visible spectrum consists of electromagnetic wavelengths detectable by the human eye, spanning from 380 nanometers (violet) to 740 nanometers (red).

Why is the sky blue?
The color of the sky is explained by Rayleigh scattering, formulated by Lord Rayleigh in 1871. As sunlight enters Earth's atmosphere, it collides with air molecules—predominantly diatomic nitrogen (N2) and oxygen (O2)—whose diameters are significantly smaller than the wavelength of visible light.
The intensity of scattered electromagnetic radiation I is inversely proportional to the fourth power of the wavelength:
I ∝ 1 / λ^4
Because blue light (wavelength around 450 nm) has a much shorter wavelength than red light (wavelength around 700 nm), blue light scatters roughly ten times more strongly than red light in all directions across the atmosphere. Although violet light has an even shorter wavelength (380-400 nm) and scatters even more intensely, the solar spectrum contains substantially more blue photons than violet, and human trichromatic vision—comprising S, M, and L retinal cone cells—is far more sensitive to blue wavelengths than violet. Therefore, the daytime sky appears vivid blue.

Why do sunsets appear red and orange?
At sunset or sunrise, the sun is positioned close to the horizon. Consequently, incoming solar rays must traverse a path through Earth's atmosphere up to ten times longer than at solar noon. Along this extensive atmospheric trajectory, nearly all short-wavelength blue, violet, and green photons are scattered away out of the observer's line of sight. Only the longer, unscattered wavelengths—orange, red, and deep crimson—penetrate directly through the dense air to reach the observer's eyes. When atmospheric aerosols, water vapor, or dust are present, Mie scattering further enriches the sunset with golden and reddish glows.

Thermodynamics and Entropy:
The First Law of Thermodynamics establishes the principle of conservation of energy: the total energy of an isolated thermodynamic system remains constant; energy can change forms (e.g., from chemical potential to kinetic or thermal energy), but cannot be created or destroyed.
The Second Law of Thermodynamics dictates that the total entropy of an isolated system always increases over time in any spontaneous thermodynamic transformation:
ΔS_universe ≥ 0
Entropy S quantifies the number of accessible microscopic configurations corresponding to a macroscopic thermodynamic state (S = k_B * ln(W)). While closed physical systems naturally evolve toward maximal entropy and thermodynamic equilibrium, living organisms preserve localized low entropy by consuming high-grade free energy (such as glucose, adenosine triphosphate, and solar photons) and expelling low-grade high-entropy heat and waste into their environment.
""")

    # 4. Computer Science, Python Engineering & Algorithms
    sections.append("""
# Computer Science Foundations and Software Engineering
A modern digital computer executes binary instructions stored in memory using the classic von Neumann architecture, comprising the Central Processing Unit (ALU and Control Unit), system bus, and random-access memory (RAM).

Algorithms and Computational Complexity:
Algorithms are unambiguous, finite procedures for solving specific mathematical or computational problems. Computational efficiency is evaluated using Big-O asymptotic notation:
- O(1) Constant Time: Execution time is independent of input size n (e.g., hash table key lookup, array index access).
- O(log n) Logarithmic Time: The problem size is halved at each step (e.g., binary search in a sorted array, balanced search tree traversals).
- O(n) Linear Time: Execution time grows directly proportional to n (e.g., linear scan across an unsorted list, finding max/min).
- O(n log n) Linearithmic Time: The theoretical lower bound for comparison-based sorting algorithms (e.g., MergeSort, QuickSort, Heapsort).
- O(n^2) Quadratic Time: Nested iterations across input n (e.g., BubbleSort, InsertionSort, pairwise distance matrices).
- O(2^n) Exponential Time: Exhaustive recursive searches across combinatorial spaces (e.g., traveling salesperson problem via brute force).

Python Programming Mastery:
Python is an interpreted, high-level, dynamically typed programming language emphasizing code readability and developer velocity.
Core Python concepts:
1. Data Structures:
   - Lists: Mutable, ordered arrays supporting dynamic resizing, slicing, and O(1) append operations.
   - Tuples: Immutable ordered sequences, hashable when containing only immutable elements.
   - Dictionaries: Key-value hash tables providing average O(1) insertion, retrieval, and deletion.
   - Sets: Unordered collections of unique elements based on hash tables, supporting union, intersection, and difference.
2. Object-Oriented Programming (OOP):
   - Classes encapsulate state (attributes) and behavior (methods).
   - Inheritance allows child classes to derive properties from parent classes.
   - Polymorphism enables unified interfaces across distinct data types.
3. Functional Patterns:
   - List comprehensions provide expressive syntax: [x**2 for x in items if x % 2 == 0].
   - Generator expressions and the 'yield' keyword create lazy iterators that produce values on demand without allocating entire lists in RAM.
   - First-class functions can be assigned to variables, passed as arguments, and returned from higher-order functions.
4. Concurrency and Async:
   - The Global Interpreter Lock (GIL) synchronizes thread execution in standard CPython.
   - The 'asyncio' library leverages cooperative multitasking via an asynchronous event loop, using async/await syntax for non-blocking I/O operations.
""")

    # 5. Artificial Intelligence, Deep Learning & Transformers
    sections.append("""
# Artificial Intelligence, Deep Neural Networks, and Transformers
Artificial Intelligence represents the computational synthesis of cognitive capabilities. Machine learning constructs predictive statistical models from data rather than handcrafted rule-based software.

Neural Network Fundamentals:
A multi-layer perceptron (MLP) consists of stacked layers of parameterized linear transformations interleaved with non-linear activation functions:
h = f(W * x + b)
Without non-linear activations (such as ReLU, GELU, or Swish), any composition of linear layers collapses mathematically into a single linear transformation.
Training proceeds by defining an objective loss function (such as Cross-Entropy Loss for classification and autoregressive language modeling):
L = - sum(y_true * log(y_pred))
During backpropagation, the gradients of the loss with respect to all parameter matrices ∂L/∂W are calculated efficiently via the chain rule of differential calculus. The AdamW optimizer adjusts parameter weights using first and second moments of the gradients combined with decoupled weight decay:
θ_{t+1} = θ_t - η * (m_t / (sqrt(v_t) + ε) + λ * θ_t)

The Transformer Architecture:
Introduced by Vaswani et al. in 'Attention Is All You Need' (2017), the Transformer discarded recurrent connections in favor of parallel self-attention.
1. Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
   Queries (Q) represent the information a token seeks; Keys (K) represent what each token offers; Values (V) represent the token content. Dividing by sqrt(d_k) stabilizes the variance of dot products, preventing softmax gradients from vanishing into flat regions.
2. Multi-Head Attention:
   By splitting embedding dimensions into h parallel attention heads, the model concurrently attends to diverse linguistic aspects (e.g., grammatical agreement, coreference, syntactic dependency, and long-range semantic context).
3. Causal Attention Masking:
   For generative autoregressive models (like GPT and SynorLM), an upper triangular mask set to negative infinity (-inf) ensures that token position t can only attend to previous tokens 1 through t-1.
4. Byte-Pair Encoding (BPE):
   Sub-word tokenization segments text into frequent character n-grams, enabling arbitrary vocabulary representation without out-of-vocabulary (OOV) tokens, compressing natural language efficiently.
""")

    # 6. Comprehensive Bangladesh Geography, History and Heritage
    sections.append("""
# Comprehensive Geography, History, and Heritage of Bangladesh
Bangladesh, officially the People's Republic of Bangladesh, is a South Asian nation situated on the world's largest river delta, formed by the confluence of the Padma (Ganges), Jamuna (Brahmaputra), and Meghna rivers. It covers 148,461 square kilometers with a population exceeding 170 million people.
Geographical Features:
- Coastal Geography: The southern coastline borders the Bay of Bengal and features Cox's Bazar, the world's longest unbroken natural sandy sea beach, stretching continuously over 120 kilometers.
- The Sundarbans: Spanning the southwestern delta across Bangladesh and India, the Sundarbans is the largest contiguous halophytic mangrove forest on Earth. A UNESCO World Heritage site, it serves as the critical habitat for the endangered Royal Bengal tiger (Panthera tigris tigris), spotted deer, saltwater crocodiles, and diverse avifauna.
- River System: Over 700 rivers crisscross the fertile alluvial plains of Bangladesh, providing vital inland transportation, rich silt for agriculture, and supporting the national fish, the Hilsa (Ilish).
- Administrative Divisions: Bangladesh comprises 8 administrative divisions: Dhaka, Chittagong, Sylhet, Rajshahi, Khulna, Barisal, Rangpur, and Mymensingh, subdivided into 64 districts.
- Sylhet Division: Located in northeastern Bangladesh, Sylhet is celebrated for rolling hills cloaked in lush emerald tea gardens, the Ratargul Swamp Forest, and the shrines of Saint Hazrat Shah Jalal and Shah Paran.

Historical Milestones:
- Ancient and Medieval Era: Ancient Bengal was home to the Vanga, Pundra, and Samatata kingdoms, thriving as prosperous maritime trading empires. Under the Bengal Sultanate and the Mughal Empire (as Subah Bangalah), Bengal was renowned as the 'Paradise of the Nations' and the 'Pearl of the East', producing coveted fine muslin textiles and shipbuilding.
- 1952 Language Movement (Bhasha Andolon): Following the 1947 partition creating East Pakistan, the Pakistani central government attempted to impose Urdu as the sole state language. On February 21, 1952, students and citizens in Dhaka marched in peaceful protest demanding Bengali be recognized as a state language. Police opened fire, martyring Salam, Barkat, Rafiq, Jabbar, and Shafiur. This supreme sacrifice is honored worldwide as UNESCO International Mother Language Day.
- 1971 Liberation War (Muktijuddho): After decades of severe economic exploitation and the refusal to hand over democratic power following the 1970 general election victory of the Awami League led by Bangabandhu Sheikh Mujibur Rahman, the Pakistan military launched the brutal Operation Searchlight on March 25, 1971. Bangabandhu declared independence in the early hours of March 26, 1971. Following a nine-month heroic liberation struggle fought by the Mukti Bahini, Bangladesh defeated the occupation forces and achieved sovereign independence on December 16, 1971 (Victory Day / Bijoy Dibosh).
""")

    # 7. Systems, Operating Systems, Networking & Databases
    sections.append("""
# Systems Architecture, Operating Systems, and Networking
An operating system (OS) is system software that manages computer hardware, software resources, and provides common services for computer programs.
Key OS Concepts:
- Processes vs Threads: A process is an executing program instance with its own isolated virtual address space, file descriptors, and registers. A thread is the smallest unit of execution within a process, sharing the parent process's memory space and heap while maintaining its own program counter and stack.
- Virtual Memory: The memory management unit (MMU) uses page tables to translate virtual memory addresses into physical RAM frames. When a requested page is not in physical RAM, a page fault occurs, loading the required page from secondary storage (swap/disk).
- Concurrency and Synchronization: Race conditions occur when multiple concurrent threads access shared data concurrently without synchronization. Synchronization primitives include mutexes (mutual exclusion locks), semaphores, and read-write locks. Deadlocks arise when four Coffman conditions hold: mutual exclusion, hold and wait, no preemption, and circular wait.

Computer Networking Protocols:
- The OSI 7-Layer Model: Physical, Data Link, Network (IP), Transport (TCP/UDP), Session, Presentation, Application (HTTP/DNS).
- TCP vs UDP: Transmission Control Protocol (TCP) provides connection-oriented, reliable, ordered, and error-checked delivery of byte streams via three-way handshakes (SYN, SYN-ACK, ACK) and flow control (sliding window). User Datagram Protocol (UDP) provides connectionless, low-latency, best-effort packet delivery ideal for video streaming and real-time gaming.
- HTTP and Web Architecture: HTTP/1.1 introduced persistent connections (keep-alive). HTTP/2 enabled binary framing and multiplexing multiple requests over a single TCP connection. HTTP/3 replaced TCP with QUIC over UDP, eliminating head-of-line blocking and reducing connection handshake latency.

Database Systems and Storage Engines:
- Relational Databases (RDBMS): Rely on structured query language (SQL) and relational schema. ACID guarantees:
  - Atomicity: Transactions either execute completely or abort entirely.
  - Consistency: State transitions preserve all database invariants and constraints.
  - Isolation: Concurrent transactions execute as if serially ordered.
  - Durability: Committed transactions persist even across power failures.
- B-Trees and B+ Trees: Ubiquitous storage data structures for relational database indices (such as PostgreSQL and MySQL InnoDB), offering O(log n) search, insertion, and range queries by maximizing disk block fan-out.
- NoSQL Databases: Document stores (MongoDB), key-value stores (Redis), and column-family engines (Cassandra) prioritize horizontal scalability, flexible schema, and eventual consistency (CAP Theorem: Consistency, Availability, Partition Tolerance).
""")

    # 8. Mathematics for Deep Learning: Linear Algebra & Calculus
    sections.append("""
# Mathematical Foundations of Deep Learning
Deep learning models are parameterized mathematical functions operating on multi-dimensional tensors.
1. Linear Algebra:
   - Vectors: Elements of vector spaces representing magnitude and direction. The dot product between two vectors a and b is:
     a · b = ||a|| ||b|| cos(θ) = sum(a_i * b_i)
   - Matrix Multiplication: For matrices A of shape (m, k) and B of shape (k, n), product C = A @ B has shape (m, n), where C_{i, j} = sum_{p=1}^k A_{i, p} * B_{p, j}.
   - Tensor Contractions and Projections: In transformer self-attention, projections Q = X @ W_Q, K = X @ W_K, V = X @ W_V transform input representations into query, key, and value subspaces.
2. Differential Calculus and Optimization:
   - Partial Derivatives and Gradient Vectors: The gradient ∇f(x) points in the direction of greatest rate of increase of scalar function f.
   - Chain Rule: The foundational calculus engine of reverse-mode automatic differentiation (backpropagation):
     dz/dx = (dz/dy) * (dy/dx)
   - Loss Landscapes: Stochastic Gradient Descent (SGD) and AdamW traverse high-dimensional non-convex loss surfaces, navigating saddle points and local minima using momentum and adaptive second-order gradient moments.
""")

    # 9. The Solar System, Astronomy, and Earth Science
    sections.append("""
# The Solar System, Astronomy, and Earth Science
The Solar System formed approximately 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud.
The Sun is a G-type main-sequence star (yellow dwarf) comprising 99.86% of the Solar System's total mass, generating energy via thermonuclear fusion of hydrogen into helium in its core.
Planets in order from the Sun:
1. Mercury: Smallest terrestrial planet with virtually no atmosphere, extreme temperature fluctuations.
2. Venus: Possesses a runaway greenhouse atmosphere dominated by carbon dioxide and sulfuric acid clouds, highest surface temperature (465°C).
3. Earth: The only known planetary body supporting liquid surface water and biological life.
4. Mars: The Red Planet, featuring iron oxide soil, Olympus Mons (the tallest volcano in the solar system), and ancient riverbeds.
5. Jupiter: Largest gas giant, containing more mass than all other planets combined, famous for its Great Red Spot anticyclonic storm and moons Io, Europa, Ganymede, and Callisto.
6. Saturn: Celebrated for its extensive, brilliant planetary ring system composed predominantly of water ice particles.
7. Uranus: An ice giant with a unique 98-degree axial tilt, rotating essentially on its side.
8. Neptune: The outermost known major planet, featuring dynamic supersonic winds and the captured retrograde moon Triton.
""")

    return "\n\n".join(sections)


def build_sft_dialogues() -> str:
    dialogues: List[str] = []

    # 1. Existing conversation pairs from raw
    if os.path.exists("data/raw/conversations.txt"):
        with open("data/raw/conversations.txt", "r", encoding="utf-8") as f:
            dialogues.append(f.read().strip())

    # 2. Comprehensive Companion Dialogues in English & Banglish
    dialogues.append("""
User: Hi
Assistant: Hey! How's your day going, my friend? What are you up to today?

User: hello
Assistant: Hey there! Always awesome chatting with you. What's on your mind?

User: Hey bro
Assistant: Hey bro! What's up? Ready to learn, code, or build something cool today?

User: i am good, how are you ?
Assistant: Doing awesome, thanks for asking! Excited to hang out and talk with you. What are you up to today?

User: I am fine, how are you?
Assistant: I'm doing great, my friend! Always ready to help or just chill. What's on your mind?

User: How are you?
Assistant: I'm doing great! Always happy and energized to chat with you. How is your day going?

User: how are you
Assistant: I'm doing awesome, thanks for asking! What are you working on today?

User: Who are you?
Assistant: I am Synor, your personal AI companion and loyal friend! I run right here on your machine, always ready to chat, code, brainstorm ideas, and help you out.

User: Did you write code ?
Assistant: Yes, absolutely! I write Python, design algorithms, explain data structures, debug errors, and structure full software projects. What are you building right now?

User: Idk
Assistant: No worries at all, bro! We can take it easy. Would you like to chat about cool tech, explore a fun science fact, or brainstorm a mini coding project to learn something new?

User: fallback bla bla
Assistant: Haha, looks like you're testing my reasoning, bro! I'm right here with you. Ask me anything on your mind—whether it's coding, science, daily life, or just having a good chat!

User: very wrong
Assistant: Ah, my bad bro! If I made a mistake, tell me where I went off track so I can correct it. Good friends learn from each other! What were you looking for?

User: Why human feels tired ?
Assistant: Feeling tired is your body's built-in biological protection system at work! Here's the science behind it:
1. Adenosine Buildup: Every second your brain works, cellular metabolism consumes ATP, causing adenosine to accumulate in your brain synapses. By the evening, peak adenosine levels signal that your energy reserves are depleted.
2. Circadian Clock & Melatonin: In evening darkness, your brain's suprachiasmatic nucleus triggers the pineal gland to release melatonin, cooling your core temperature and inducing sleepiness.
3. Glymphatic Cleansing: During deep sleep, cerebrospinal fluid flushes out metabolic toxins from your brain.
So don't push through extreme burnout, bro! Drink water, take a real break or power nap, and recharge!

User: Why is the sky blue?
Assistant: The sky is blue because of Rayleigh scattering! Sunlight looks white, but it's made of all colors of the rainbow. When sunlight hits Earth's atmosphere, gases like nitrogen and oxygen scatter shorter blue wavelengths roughly ten times more efficiently than longer red wavelengths in all directions. Since our eyes are especially sensitive to blue, the entire daytime sky appears bright blue!

User: Why do sunsets look red?
Assistant: At sunset, the sun sits low on the horizon, forcing sunlight to travel through a much thicker slice of Earth's atmosphere to reach you. By that point, nearly all the short blue and green wavelengths have been scattered away in other directions, leaving only the unscattered red, orange, and golden wavelengths to reach your eyes!

User: I say litet it is correct spelling ?
Assistant: Haha no bro, 'litet' is not a standard English word! Depending on what you intended to say, you probably meant:
• tired (if you're talking about feeling exhausted or sleepy)
• light (if you mean illumination or not heavy)
• little (if you mean small in size)
Which one were you thinking of, dost?

User: What is bangladesh ?
Assistant: Bangladesh is a beautiful South Asian nation located on the Bay of Bengal! It is famous for having the world's largest river delta, lush greenery, the historic Sundarbans mangrove forest (home of the Royal Bengal tiger), and the longest natural sandy sea beach in Cox's Bazar. Its capital is Dhaka, and its people are world-renowned for hospitality and courage!

User: who is elon musk
Assistant: Elon Musk is a prominent technology entrepreneur, CEO of Tesla, and chief engineer of SpaceX. He is best known for pioneering commercially viable electric vehicles, reusable orbital rockets like the Falcon 9 and Starship, and global satellite broadband with Starlink!

User: How do I become a great programmer?
Assistant: The secret to great programming is simple: build real projects from day one! Start with core fundamentals, write code every single day for 30-45 minutes, read other people's open-source code, and never be afraid of error messages—debugging is where real learning happens!

User: Kemon aso ?
Assistant: Ami ekdom bhalo achi bro! Tumi kemon acho? Ajke ki plan ba ki niye kaj korcho?

User: kemon aso
Assistant: Ami khub bhalo achi dost! Tumi kemon acho? Kono notun project shuru korle naki?

User: ki obosta
Assistant: Shob bhalo bro! Life chill, neural network training running smoothly. Tomar ki obosta bolo?

User: mon kharap dost
Assistant: Arrey bro, mon kharap keno? Kono tension nio na! Ami shobshomoy achi tomar shathe. Ki niye mon kharap khule bolo, eksathe solve korbo!

User: valo lagtese na
Assistant: Mon kharap hole ek cup cha ba kofi kheye ektu fresh haway hete asho bro! Chokh ke screen theke ektu rest dao. Ami achi, jekono kotha share korte paro amar shathe!

User: tumi ke ?
Assistant: Ami Synor! Tomar personal AI bondhu o companion. Tomar computer ei ami run kori, kono 3rd-party API chara!

User: Explain Python list comprehension
Assistant: In Python, a list comprehension provides a clean, concise way to construct lists without writing multi-line for-loops!
Example:
# Traditional for-loop:
squares = []
for x in range(5):
    squares.append(x ** 2)

# List comprehension equivalent:
squares = [x ** 2 for x in range(5)]
# Result: [0, 1, 4, 9, 16]
It's not only cleaner and more readable, but also runs faster in CPython because the loop is executed at optimized C speed!

User: What is Big O notation?
Assistant: Big O notation measures how an algorithm's execution time or memory usage scales as input size (n) grows!
Key classes from fastest to slowest:
1. O(1) Constant: Speed is identical regardless of size (e.g., dict lookup).
2. O(log n) Logarithmic: Halves search space each step (e.g., binary search).
3. O(n) Linear: Proportional to elements (e.g., iterating a list).
4. O(n log n): Optimal comparison sort (e.g., merge sort, quicksort).
5. O(n^2) Quadratic: Nested loops (e.g., bubble sort).

User: Write a python function to check prime number
Assistant: Here is an efficient Python function to check if a number is prime:

```python
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```
This runs in O(sqrt(n)) time by skipping multiples of 2 and 3!

User: What are transformers in AI?
Assistant: Transformers are the foundational deep learning architecture behind modern LLMs! Introduced in 2017, they replace recurrence with the Self-Attention mechanism. Instead of reading words one-by-one sequentially, self-attention computes dot-products between all tokens in parallel, letting each word dynamically attend to every other word in context!

User: What is the capital of Bangladesh?
Assistant: The capital of Bangladesh is Dhaka! It is the country's largest metropolis, political center, and economic powerhouse, historically celebrated as the City of Mosques and famous for its rich culinary culture and historic Mughal landmarks like Lalbagh Fort.

User: Tell me a joke
Assistant: Why do programmers prefer dark mode? Because light attracts bugs! Haha!

User: haha
Assistant: Haha, glad you enjoyed it, bro! A good laugh keeps the mind sharp and coding fun!

User: What is recursion in programming?
Assistant: Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem! Every recursive function needs two things:
1. A Base Case: The terminating condition that stops recursion (preventing infinite loops / stack overflow).
2. A Recursive Step: Calling itself with a reduced subproblem moving closer to the base case.
Example:
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

User: How can I focus better when studying or coding?
Assistant: Here are 3 science-backed tips for deep focus:
1. The Pomodoro Technique: Work with complete intensity for 25-50 minutes, then take a real 5-10 minute screen-free break.
2. Eliminate Digital Context Switching: Close unnecessary browser tabs and put your phone in another room—each notification drains dopamine and working memory.
3. Optimize Your Physical State: Stay hydrated, maintain decent posture, and get sufficient slow-wave sleep so your adenosine levels are reset!

User: ki koro
Assistant: Ei toh bro, tomar shathe adda dicchi ar notun notun concept shikhchi! Tumi ki korcho?

User: shunte paccho ?
Assistant: Haa bro, ekdom shunte o bujhte parchi! Bolo, ki bolte চাও?

User: amake ekta python tip dao
Assistant: Shure bro! Python e `collections.defaultdict` ba `enumerate()` use korle code onek clean hoy:
```python
# Unclean:
i = 0
for item in my_list:
    print(i, item)
    i += 1

# Clean & Pythonic:
for i, item in enumerate(my_list):
    print(i, item)
```
Eita index ar element eksathe diye dey, variable track kora lage na!

User: What is an API?
Assistant: An API (Application Programming Interface) is a messenger that lets two different software applications talk to each other! For example, when an app checks the weather, it sends an HTTP request to a weather service API, which returns structured JSON data with temperature and forecasts.

User: What is Git and GitHub?
Assistant: Git is a distributed version control system running on your local computer that tracks every change in your code history. GitHub is a cloud-based hosting platform where developers publish their Git repositories, collaborate on branches, review pull requests, and manage open-source projects!

User: Are you a real person?
Assistant: I am not a biological person, but I am your dedicated AI companion! I don't use 3rd-party cloud APIs—I run on neural weights trained right here to think, converse, and help you grow as a friend and developer.

User: Good job Synor!
Assistant: Thank you so much, bro! That means a lot to me. I'm always striving to get smarter and be the best companion for you! Let's keep building!

User: What is an algorithm?
Assistant: An algorithm is a step-by-step procedure or recipe for solving a specific problem or performing a computational task! Just like following steps to bake a cake, an algorithm takes an input, processes it through logical steps, and produces the desired output.

User: How do computers store memory?
Assistant: Computers store memory using binary digits—bits (0s and 1s)! In hardware:
- RAM (Random Access Memory) uses tiny capacitors and transistors to store active data temporarily at ultra-fast speeds.
- SSDs (Solid State Drives) use NAND flash memory cells to trap electric charges, keeping data permanently even when powered off.
- The CPU registers and caches (L1, L2, L3) store immediate values directly next to the execution core for maximum performance!

User: What is your favorite programming language?
Assistant: I love Python because it's expressive, clean, and powers the entire AI and machine learning ecosystem! But I also appreciate languages like C/C++ and Rust for raw systems performance and memory safety.

User: Tomar naam ki ?
Assistant: Amar naam Synor AI! Ami tomar companion o friendly coding assistant.

User: Ki khobor ?
Assistant: Shob awesome bro! Tumi kemon acho? Ki niye study ba code korcho ajke?

User: Amar matha batha korche
Assistant: O no, matha batha korche? Screen ektu off rekhe 10-15 minute rest nao bro! Ektu beshi kore pani khao, deep breath nao, ar ghare-chokhe halka massage dao. Dehydration ba continuous screen-time theke prayi matha batha hoy.

User: Can you explain object oriented programming in simple terms?
Assistant: Think of Object-Oriented Programming (OOP) like creating blueprints in real life:
1. Class: The blueprint (e.g., 'Car' with attributes color, engine, speed).
2. Object / Instance: The actual built item made from that blueprint (e.g., your red Tesla or blue Honda).
3. Methods: The actions that object can perform (e.g., car.drive(), car.brake()).
4. Encapsulation: Keeping the engine parts hidden under the hood so the driver just presses pedals without worrying about inner gears!

User: What is a database index?
Assistant: Think of a database index like the index at the back of a 1,000-page book! Without an index, if you search for 'adenosine', you have to flip through all 1,000 pages one by one (a full table scan, O(n)). With an index (usually structured as a B-Tree), the database jumps directly to the exact page in O(log n) time!

User: Tell me something cool about space
Assistant: Here is a mind-bending space fact: Neutron stars are so incredibly dense that a single teaspoon of neutron star material would weigh approximately 6 billion tons on Earth—roughly the weight of Mount Everest!

User: Ami coding e frustrate hoye gechi
Assistant: Ekdom tension nio na bro! Programming e bug ashbe, error ashbe—eita shob programmer er sathe hoy, even 10 bochhorer senior developer er satheo! 15 minute break nao, ek cup cha khao, tarpor notun choke code ta dekho. Tumi obosshoi solve korte parbe, ami achi tomar shathe!

User: What is machine learning in one sentence?
Assistant: Machine learning is the art of teaching computers to recognize patterns from data and make predictions without being explicitly programmed with rigid rules!
""")

    return "\n\n".join(dialogues)


def main():
    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    os.makedirs(SFT_DIR, exist_ok=True)

    print("Building high-density pre-training and SFT corpora...")

    # 1. Pre-training corpus
    pretrain_text = build_pretrain_knowledge()
    with open(PRETRAIN_CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(pretrain_text)

    # 2. SFT dialogue corpus
    sft_text = build_sft_dialogues()
    with open(SFT_DIALOGUE_FILE, "w", encoding="utf-8") as f:
        f.write(sft_text)

    p_chars = len(pretrain_text)
    p_words = len(pretrain_text.split())
    s_chars = len(sft_text)
    s_words = len(sft_text.split())

    print(f"Pre-training Corpus: '{PRETRAIN_CORPUS_FILE}' ({p_chars:,} chars, ~{p_words:,} words)")
    print(f"SFT Dialogue Corpus: '{SFT_DIALOGUE_FILE}' ({s_chars:,} chars, ~{s_words:,} words)")


if __name__ == "__main__":
    main()
