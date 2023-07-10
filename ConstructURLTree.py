from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
import pickle
import random
import sys


#define a tree node with a token value and a list of children
class Node:
    def __init__(self, token):
        self.token = token
        self.children = {}
        self.isLeaf = True
        self.isPopulated = False
        self.leafCompletion = []

    def Insert(self, sequence):
        if self.isLeaf:
            if not self.isPopulated:
                self.leafCompletion = sequence
                self.isPopulated = True
            else:
                # No longer a leaf, we need to insert the leaf completion into the tree, then insert ourselves
                self.isLeaf = False
                self.Insert(self.leafCompletion)
                self.Insert(sequence)
        else:
            # We are not a leaf, so we need to insert ourselves into the tree
            if len(sequence) > 0:
                # Split off the first token to use as a key
                insertToken = sequence[0]
                # Then insert the rest of the sequence at that key
                if insertToken not in self.children:
                    self.children[insertToken] = Node(insertToken)
                self.children[insertToken].Insert(sequence[1:])                
            else:
                # We have no more tokens to insert, so we are done
                return
    def CheckChildExists(self, token):
        return token in self.children
    def CheckIsLeaf(self):
        return self.isLeaf
    def GetLeafCompletion(self):
        return self.leafCompletion
    def GetRandomChild(self):
        return random.choice(list(self.children.values()))

root = Node(-1)


numurls = 0
# Grab the url list from the file
with open('wiki_urls.txt', 'r') as f:
    urls = f.readlines()

    numurls = len(urls)

    if False: #Control whether we use all the urls or just a subset
        random.shuffle(urls)
        urls = urls[:100000]

    # Loop through all the urls and prepend en.wikipedia.org/wiki/ to them
    for i in range(len(urls)):
        urls[i] = "en.wikipedia.org/wiki/" + urls[i]
        # Let's strip newlines, while we're at it, because I'm sure those things slipped in there somewhere
        urls[i] = urls[i].rstrip('\n')

    # Tokenize the urls
    tokenizedURLS = []
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

    eos_token_id = tokenizer.eos_token_id

    # Loop through the urls and tokenize them
    for url in urls:
        tokenizedURLS.append(tokenizer.encode(url, return_tensors="pt"))
        #show progress
        if len(tokenizedURLS) % 10000 == 0:
            print("Tokenized " + str(len(tokenizedURLS)) + " of " + str(numurls) + " urls.")    
    
    #convert the tokenized urls to lists of tokens
    tokenlists = []

    #Loop through all the tokenized urls, and convert them to lists
    for turl in tokenizedURLS:
        tlist = turl.tolist()[0]
        tlist.append(eos_token_id)
        tokenlists.append(tlist)
        #show progress
        if len(tokenlists) % 10000 == 0:
            print("Converted " + str(len(tokenlists)) + " of " + str(numurls) + " urls to lists of tokens.")

    # Add each list to our tree
    ctr = 0
    for tokenlist in tokenlists:           
        ctr += 1
        if ctr % 10000 == 0:
            print("Constructed " + str(ctr) + " of " + str(numurls) + " trees.")
        root.Insert(tokenlist)

    # Let's get a random path through the tree, detokenize it, and print it
    current = root
    path = []
    while current != None:
        if current.token > 0:
            path.append(current.token)
        if not current.isLeaf:
            current = current.GetRandomChild()
        else:
            for token in current.GetLeafCompletion():
                path.append(token)
            current = None
    print(path)
    pathTensor = torch.tensor(path)
    print(tokenizer.decode(pathTensor))
    
    #on the next line, we save the tree to a file using pickle
    sys.setrecursionlimit(10000)
    print ("Saving tree to file...")
    pickle.dump(root, open("bigtree.p", "wb"))

    #let's load that tree back in
    print ("Loading tree from file...")
    root = pickle.load(open("bigtree.p", "rb"))

    #let's get another path
    current = root
    path = []
    while current != None:
        if current.token > 0:
            path.append(current.token)
        if not current.isLeaf:
            current = current.GetRandomChild()
        else:
            for token in current.GetLeafCompletion():
                path.append(token)
            current = None
    print(path)
    pathTensor = torch.tensor(path)
    print(tokenizer.decode(pathTensor))


        
