
def groupAnagrams(strs):
    strsMap = {}
    for word in strs:
        for letter in word:
            wordMap = {}
            if letter in wordMap:
                wordMap[letter] += 1
            else: 
                wordMap[letter] = 1
            strsMap[word] = wordMap
    return strsMap

strs=["act","pots","tops","cat","stop","hat"]

print(groupAnagrams(strs))
print(ord("s"))



        