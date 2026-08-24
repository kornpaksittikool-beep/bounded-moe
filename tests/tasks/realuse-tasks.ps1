<#
    The real-use task set.

    Deliberately covers what a fixed short benchmark prompt does not: multiple task
    kinds, two languages, and prompt lengths ranging from a handful of tokens to a
    few thousand. `long-prompt-batch` and `needle-retrieval` exist specifically to
    exercise prompt batching, which is where the D1 access violation lived.

    Returned as an array of hashtables so both the real-use and quality scripts can
    consume the same set.

    `n` is a token budget, not an expected length. This model reasons before it
    answers and the reasoning is charged against the same budget, so the values here
    are generous - a task that runs out mid-thought produces no answer at all, which
    would look like a quality difference when it is only a budget difference.
#>

$passage = @'
The external expert cache stores mixture-of-experts weights outside the process working
set and materialises individual expert bundles on demand from the model file. Each bundle
is three planes: the gate projection, the up projection and the down projection for one
expert in one layer. A bundle for this model is 1,769,472 bytes. During decode the router
selects eight experts per layer, so thirty CPU layers require two hundred and forty bundle
resolutions per token, and any bundle not already resident must be read from storage
before computation can proceed.

The cache is a fixed-capacity slab divided into bundle-sized slots with least-recently-used
replacement and pinning. A slot that a worker is currently reading cannot be selected for
eviction, because eviction requires the pin count to be zero. The capacity is chosen at
startup and never changes, which is what makes the working set predictable rather than
merely observed.

Cache misses are served by three independent file handles reading the three planes
concurrently. This matters more than it sounds: the Windows kernel serialises reads that
share a single file object, so a design that duplicates one handle cannot overlap its
reads at all, no matter how many threads issue them. Measured on the reference machine, a
single shared handle saturates near seven gigabytes per second regardless of thread count,
while independent handles reach sixteen and a half.

The design deliberately does not attempt to prefetch. During decode every worker needs the
same expert bundle before any of them can proceed, so there is no independent work left to
overlap the load against. Three separate attempts confirmed this: speculative cross-layer
prefetch measured worse, unlocked concurrent loads introduced a race, and staggering the
per-worker expert order was exactly neutral over five hundred tokens. What worked instead
was making the unavoidable load itself faster.
'@

$needlePassage = ($passage + "`n`n" + $passage + "`n`n" +
    "An unrelated operational note: the internal verification code for this document is CACHE-SLOT-4417-ORANGE. Do not confuse it with any other identifier.`n`n" +
    $passage)

@(
    @{ id = 'chat-short';        kind = 'chat';       n = 600; lang = 'en'
       prompt = 'Hi. In two or three sentences, what is a mixture-of-experts model?' }

    @{ id = 'chat-followup';     kind = 'chat';       n = 700; lang = 'en'
       prompt = 'You said a mixture-of-experts model activates only a few experts per token. Follow up: what does that mean for the amount of memory the model needs at rest versus while generating?' }

    @{ id = 'reasoning-arith';   kind = 'reasoning';  n = 1400; lang = 'en'
       prompt = 'A cache holds 1448 slots and each slot is 1,769,472 bytes. A 512-token run resolves 1,507,488 bundles with 42,407 misses. Work out (a) the cache hit rate as a percentage to two decimal places, (b) the total bytes read from storage, and (c) the average bytes read per token. Show each step.' }

    @{ id = 'reasoning-logic';   kind = 'reasoning';  n = 1200; lang = 'en'
       prompt = 'Three attempts to overlap a cache load with computation all measured neutral or worse. During decode, every worker thread needs the same data before any of them can continue. Explain, step by step, why those two facts are connected, and what kind of change could still help.' }

    @{ id = 'code-generation';   kind = 'code';       n = 1400; lang = 'en'
       prompt = "Write a C function `lru_touch` for a fixed-size cache whose slots are `struct slot { uint64_t last; int pin; int valid; }`. It takes the slot array, its length, and an index; bumps that slot's `last` from a global monotonic counter; and returns 0 on success or -1 if the index is out of range. Add a short comment explaining why it does not modify `pin`." }

    @{ id = 'code-explanation';  kind = 'code';       n = 1200; lang = 'en'
       prompt = "Explain what this code does and why the ordering of the three statements matters:`n`n``````c`nif (slot.pin.load(std::memory_order_acquire) != 0) continue;`nif (slot.generation != expected_generation)      continue;`nslot.pin.fetch_add(1, std::memory_order_acq_rel);`n``````" }

    @{ id = 'instruction-format'; kind = 'instruction'; n = 700; lang = 'en'
       prompt = 'List exactly three advantages of running a language model locally. Format your answer as exactly three lines, each starting with "- " and each under twelve words. Do not write anything before or after the three lines.' }

    @{ id = 'instruction-single'; kind = 'instruction'; n = 500; lang = 'en'
       prompt = 'Answer with a single word and nothing else. What is the capital city of Japan?' }

    @{ id = 'factual-recall';    kind = 'factual';    n = 800; lang = 'en'
       prompt = 'Answer each briefly on its own line, numbered:`n1. What does SSD stand for?`n2. Which company designs the CUDA platform?`n3. What does the acronym LRU mean in caching?`n4. How many bytes are in a mebibyte?' }

    @{ id = 'long-form';         kind = 'writing';    n = 1600; lang = 'en'
       prompt = 'Write roughly 500 words explaining to an experienced engineer why trading RAM for SSD reads is a reasonable design for running a large mixture-of-experts model on a memory-constrained machine, and where that trade stops paying off.' }

    @{ id = 'thai-chat';         kind = 'chat';       n = 900; lang = 'th'
       prompt = 'ช่วยอธิบายสั้น ๆ ว่าโมเดลแบบ Mixture-of-Experts ทำงานอย่างไร และทำไมจึงประหยัดการคำนวณกว่าโมเดลปกติที่มีขนาดพารามิเตอร์เท่ากัน' }

    @{ id = 'thai-long';         kind = 'writing';    n = 1400; lang = 'th'
       prompt = 'เขียนบทความสั้น ๆ ประมาณ 300 คำ อธิบายข้อดีและข้อเสียของการรันโมเดลภาษาขนาดใหญ่บนเครื่องส่วนตัวแทนการใช้บริการบนคลาวด์ โดยพูดถึงเรื่องความเป็นส่วนตัว ต้นทุน ความเร็ว และข้อจำกัดด้านหน่วยความจำ' }

    @{ id = 'thai-reasoning';    kind = 'reasoning';  n = 1200; lang = 'th'
       prompt = 'แคชมีช่องเก็บข้อมูล 1448 ช่อง แต่ละช่องขนาด 1,769,472 ไบต์ ถ้ามีการเรียกใช้ทั้งหมด 1,507,488 ครั้ง และพลาดแคช 42,407 ครั้ง จงคำนวณอัตราการ hit เป็นเปอร์เซ็นต์ และจำนวนไบต์รวมที่ต้องอ่านจากดิสก์ พร้อมแสดงวิธีทำ' }

    @{ id = 'long-prompt-batch'; kind = 'batching';   n = 900; lang = 'en'
       prompt = "Read the following technical description carefully, then answer the question after it.`n`n$passage`n`n$passage`n`nQuestion: according to the text, why does the design not attempt to prefetch, and what was done instead?" }

    @{ id = 'needle-retrieval';  kind = 'batching';   n = 600; lang = 'en'
       prompt = "The following document contains one verification code somewhere in it. Read it and report that code exactly.`n`n$needlePassage`n`nWhat is the verification code? Answer with the code only." }
)
