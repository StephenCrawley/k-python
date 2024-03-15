# K Interpreter in Python

This Python script is a simple interpreter for a subset of the K programming language. It's designed for practice and learning purposes, providing a basic implementation of fundamental K operations.

### Usage
1. Clone the repository.
2. Run the Python script.
3. Enter K expressions to observe the results.

### Supported Operations
| Operation         | Spelling         | Result
| ----------------- | ---------------- |-----------
| Addition          | `x + y`          | `1 + 2 -> 3`   
| Multiplication    | `x * y`          | `2 * 3 -> 6`  
| Enumeration       | `!x`             | `!3 -> 0 1 2` 
| Join              | `x,y`            | `1,2 -> 1 2`  
| Box/Enlist        | `,x`             | `,1  -> ,1`   
| Reverse           | `|x `            | `|!3 -> 2 1 0`
| Top/First         | `*x `            | `*!3 -> 0`    
| Over/Reduce       | `+/x`            | `+/!5 -> 10`  
|                   | `x+/y`           | `10+/!5 -> 20`
| Scan/Accumulate   | `+\x`            | `+\!5` -> `0 1 3 6 10`
|                   | `x+\y`           | `10+\!5` -> `10 11 13 16 20`

```
/ Some simple arithmetic:

 3*2+1
9

/ Note that evaluation is right-to-left; there is no operator precedence. Parens alter evaluation order:

 (3*2)+1
7

/ We can create flat integer lists like this:

 1,2,3
1 2 3
/ or
 (1;2;3)
 1 2 3

/ And nested lists like this:

 (1,2;3,4)
(1 2;3 4)

/ Of course, arithmetic ops are fully atomic:

 (2,5) * (1,2;3,4)
(2 4;15 20)
```

## Note
This script was created for personal practice and is shared for demonstrative purposes only. Usage may be hazourdous. You've been warned!

Feel free to explore and modify the code for further learning and improvement.


