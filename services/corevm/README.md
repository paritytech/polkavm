# CoreVM Execution Model: Segment-Based Execution

## Overview

CoreVM demonstrates **segment-based memory allocation** for **Child VMs**, enabling the execution of **guest programs**. The system initializes memory using **import segments**, executes computations, and exports results as **export segments**, which then serve as input for the next cycle.

Both **Refine** and **Accumulate** extensively utilize host functions. **Refine** focuses on demonstrating CoreVM's core functionality, while **Accumulate** primarily serves to verify host function behavior.

---

# Refine: Page Data Structure

A **Page Data Segment** is a **4 KB memory page** in a **Child VM running within CoreVM**.

## Structure Overview

Each segment consists of **4104 bytes**:

| Component      | Size (Bytes) | Description                      |
| -------------- | ------------ | -------------------------------- |
| **Machine ID** | 4            | Identifies the segment’s origin. |
| **Page ID**    | 4            | Identifies the page in CoreVM.   |
| **FIB Data**   | 4            | Stores Fibonacci metadata.       |
| **Zero Bytes** | 4092         | Reserved for padding/extensions. |

- **FIB Data** is in **bytes 8–12** of the segment, extracted from the first **4 bytes of the page**.
- **Pages are dynamically imported/exported** to maintain **stateless execution**.

---

# Refine: Page Model (Import & Export Segment)

This table shows **memory page flow** in CoreVM. **Child VMs import pages for execution** and **export updated pages as output**.

| n/pages | Imported Pages                | Exported Pages                |
|---------|--------------------------------|--------------------------------|
| 0       |                                | page_0                         |
| 1       | page_0                         | page_0, page_1                 |
| 2       | page_0, page_1                 | page_0, page_1, page_2         |
| 3       | page_0, page_1, page_2         | page_0, page_1, page_2, page_3 |
| 4       | page_0, page_1, page_2, page_3 | page_0, page_1, page_2, page_3, page_4 |
| 5       | page_0, page_1, page_2, page_3, page_4 | page_0, page_1, page_2, page_3, page_4, page_5 |
| 6       | page_0, page_1, page_2, page_3, page_4, page_5 | page_0, page_1, page_2, page_3, page_4, page_5, page_6 |
| 7       | page_0, page_1, page_2, page_3, page_4, page_5, page_6 | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7 |
| 8       | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7 | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8 |
| 9       | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8 | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9 |
| 10      | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9 | page_0, page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9, page_10 |

- **Imported Pages**: Initial memory state before execution.
- **Exported Pages**: Updated memory state after execution.
- **Each page is 4 KB**, ensuring stateless execution.

---

# Refine: Fibonacci Computation Model

A **guest program** running on CoreVM performs **Fibonacci calculations** by **importing/exporting memory pages** sequentially.

| n  | FIB(n)        | FIB(n-1)              | FIB(n-2)              |
| -- | ------------- | --------------------- | --------------------- |
| 0  | 0  (page\_0)  | 0  (default\_0\_page) | 0  (default\_0\_page) |
| 1  | 1  (page\_1)  | 1  (default\_1\_page) | 0  (default\_0\_page) |
| 2  | 1  (page\_2)  | 1  (page\_1)          | 0  (page\_0)          |
| 3  | 2  (page\_3)  | 1  (page\_2)          | 1  (page\_1)          |
| 4  | 3  (page\_4)  | 2  (page\_3)          | 1  (page\_2)          |
| 5  | 5  (page\_5)  | 3  (page\_4)          | 2  (page\_3)          |
| 6  | 8  (page\_6)  | 5  (page\_5)          | 3  (page\_4)          |
| 7  | 13 (page\_7)  | 8  (page\_6)          | 5  (page\_5)          |
| 8  | 21 (page\_8)  | 13 (page\_7)          | 8  (page\_6)          |
| 9  | 34 (page\_9)  | 21 (page\_8)          | 13 (page\_7)          |
| 10 | 55 (page\_10) | 34 (page\_9)          | 21 (page\_8)          |

- **FIB Data is stored in bytes 8–12** of each segment.
- **Each page represents a memory state in CoreVM**, evolving over multiple work packages.
- Pages are **exported at each step** and **imported for the next**, ensuring continuous computation.

---

# Accumulate: Host Function Testing

Accumulate extensively tests **Accumulate host functions** by invoking different host calls based on `n`.

There are two key, value pairs:
- Key: jam, Value: DOT
- Key: DOT, Value: jam

The table below contains the host function calls and their respective results.

| n  | EP in                 | Key 0                           | Key 1                                  | Key 2                          | Key 5                  | Key 6                    | Key 7                | Key 8       | Key 9       |
|----|----------------------|--------------------------------|----------------------------------------|--------------------------------|------------------------|--------------------------|----------------------|------------|------------|
| 0  |                      | FIB result_0                   |                                        |                                |                        |                          |                      | info OK    | gas result |
| 1  |                      | FIB result_1                   | read jam NONE                          | write jam NONE                 | read jam (OK:3)        | forget HUH               |                      | info OK    | gas result |
| 2  |                      | FIB result_2                   | read jam (OK:3)                        | write deleted jam OK:3         | read jam NONE          |                          |                      | info OK    | gas result |
| 3  |                      | FIB result_3                   | solicit hash("jam") (OK)               | query hash("jam") (OK:0)       | query hash("dot") NONE |                          |                      | info OK    | gas result |
| 4  | EP ("jam") in, (insert t) | FIB result_4             | forget hash("jam") (OK, insert t)      | query hash("jam") (OK:2+2^32*x) | lookup hash("dot") NONE | assign CORE              |                      | info OK    | gas result |
| 5  |                      | FIB result_5                   | lookup hash("jam") (OK:3)              | query hash("jam") (OK:2+2^32*x) | eject WHO              | bless WHO                |                      | info OK    | gas result |
| 6  |                      | FIB result_6                   | solicit hash("jam") (OK) (insert t)    | query hash("jam") (OK:3+2^32*x) | assign OK              |                          |                      | info OK    | gas result |
| 7  | EP ("jam") in        | FIB result_7                   | forget hash("jam") (OK) (insert w,t)          | query hash("jam") (OK:2+2^32*x) |                        |                          |                      | info OK    | gas result |
| 8  |                      | FIB result_8                   | lookup hash("jam") (OK:3)              | query hash("jam") (OK:2+2^32*x) |                        |                          |                      | info OK    | gas result |
| 9  |                      | FIB result_9                   | new (OK: service index) (checkpointed, same as previous) | upgrade OK (checkpointed, same as previous)   | bless OK (checkpointed, same as previous)   |                          |                      | info OK    | gas result |
| 10 |                      | FIB result_10                  | write delete DOT NONE                  | write DOT NONE                 | write delete DOT OK:3  | read DOT NONE            | write delete DOT NONE | info OK    | gas result |

