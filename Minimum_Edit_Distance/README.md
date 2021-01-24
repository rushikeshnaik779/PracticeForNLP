## MINIMUM EDIT DISTANCE

- Minimum Edit distance between two strings str1 and str2 is defined as the minimum number of insert/delete/substitute operations required to transform str1 into str2. For example if str1 = "ab", str2 = "abc" then making an insert operation of character 'c' on str1 transforms str1 into str2. Therefore, edit distance between str1 and str2 is 1. You can also calculate edit distance as number of operations required to transform str2 into str1. For above example, if we perform a delete operation of character 'c' on str2, it is transformed into str1 resulting in same edit distance of 1.

- Looking at another example, if str1 = "INTENTION" and str2 = "EXECUTION", then the minimum edit distance between str1 and str2 turns out to be 5 as shown below. All operations are performed on str1.\
![IMAGE OF TASK](https://github.com/rushikeshnaik779/PracticeForNLP/blob/main/Minimum_Edit_Distance/Screenshot%202021-01-24%20at%208.55.08%20AM.png)

IMPLEMENTED EXAMPLE: 
![IMAGE OF IMPLEMENTATION](https://github.com/rushikeshnaik779/PracticeForNLP/blob/main/Minimum_Edit_Distance/Screenshot%202021-01-24%20at%208.55.54%20AM.png)
