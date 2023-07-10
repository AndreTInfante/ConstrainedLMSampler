from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
import pickle

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

class ConstrainedSampler:
    def __init__(self, model_id="tiiuae/falcon-7b-instruct"):
        # Check for CUDA support
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print ("Using device:", self.device)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)  # Move the model to GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Model loaded.")

    def sample(self, sequence, root, num_tokens_to_generate=100):
        
        input_ids = self.tokenizer.encode(sequence, return_tensors="pt").to(self.device)
        completion_ids = self.tokenizer.encode(" ", return_tensors="pt").to(self.device)
        current = root

        top_k = 1 # This forces purely greedy sampling, which seems to work noticeably better
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(num_tokens_to_generate):

            sequence_ids = torch.cat((input_ids, completion_ids), dim=1)

            with torch.no_grad():
                outputs = self.model(sequence_ids)
            
            predictions = outputs.logits[:, -1, :]

            probabilities = F.softmax(predictions, dim=-1)

            # Convert probabilities to a list of tuples of (probability, token_id)
            # TO-DO: There must be a better way to do this
            probabilities = [(probabilities[0][i].item(), i) for i in range(len(probabilities[0]))]

            # Exclude all the tokens that aren't in current.children
            probabilities = [prob for prob in probabilities if prob[1] in current.children]

            if len(probabilities) == 0:
                break
            
            # Reassemble our probabilities
            probs, token_ids = zip(*probabilities)
            probs = torch.tensor(probs).to(self.device)
            token_ids = torch.tensor(token_ids).to(self.device)

            # Do sampling normally on our reduced list
            new_top_k = min(top_k, len(probabilities))
            top_k_probs, top_k_ids = torch.topk(probs, new_top_k)
            new_token = torch.multinomial(top_k_probs, num_samples=1)
            new_token_id = token_ids[top_k_ids[new_token]]
            new_token_id = new_token_id.unsqueeze(0)

            #explore the tree in the direction we've chosen
            current = current.children[new_token_id.item()]

            completion_ids = torch.cat((completion_ids, new_token_id), dim=1)

            #print periodically so it's clear it's still working
            if (len(completion_ids[0]) + 1) % 9 == 0:
                print("Thinking...")

            if new_token_id == eos_token_id:
                break

            if(current.isLeaf and current.isPopulated):
                #Add the leaf completion to the completion_ids
                for token in current.GetLeafCompletion():
                    #turn the token into a tensor
                    tt = torch.tensor([token]).to(self.device)
                    completion_ids = torch.cat((completion_ids, tt.unsqueeze(0)), dim=1)
                break

        # Drop the last token from the completion_ids, as it's just an eos token
        completion_ids = completion_ids[:, :-1]
        result = self.tokenizer.decode(completion_ids[0])

        # Bad torch, no memory leaks
        del input_ids, completion_ids, outputs, predictions, probabilities, probs, token_ids, top_k_probs, top_k_ids, new_token, new_token_id
        torch.cuda.synchronize()
        return result


sampler = ConstrainedSampler()

#let's load that tree back in
print ("Loading tree from file. This may take a while, pickle hurries for nobody.")
# Big tree is every en.wikipedia url, small tree is a subset of 100,000 random urls for faster testing
root = pickle.load(open("bigtree.p", "rb"))
print ("Tree loaded.")

prompts = []

#add some test questions
prompts.append("What is the capital of the United States?")
prompts.append("What is the capital of France?")
prompts.append("Who sang Bohemian Rhapsody?")
prompts.append("What mineral is emerald made of?")
prompts.append("Who shot Abraham Lincoln?")
prompts.append("Who wrote the book The Three Musketeers?")
prompts.append("What chemical elements are diamonds made of?")
prompts.append("Who was the first human in space?")

prefix = "Question: "
postfix = ". Reply with only the URL of the wikipedia page that best answers this question. Answer: "

#go through and test each prompt and print the result
for prompt in prompts:
    print("Prompt: " + prompt)
    prompt = prefix + prompt + postfix
    print("Result: " + sampler.sample(prompt, root))

# Let the user ask interactive questions
while True:
    prompt = input("Enter a question: ")
    #strip newline from prompt
    prompt = prompt.rstrip('\n')
    prompt = prefix + prompt + postfix
    print("Answer: " + sampler.sample(prompt, root))

