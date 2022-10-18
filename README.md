# **ZymCTRL**

ZymCTRL ([preprint coming soon](https://huggingface.co/nferruz/ZymCTRL)) is a conditional language model trained on the BRENDA database of enzymes. Given a user-defined Enzymatic Commission (EC) number, the model generates protein sequences that fulfill that catalytic reaction. The generated sequences are ordered, globular and distant to natural ones, while their intended catalytic properties match those defined by users.



## **Model description**
ZymCTRL is based on the CTRL Transformer architecture and contains 36 layers with a model dimensionality of 1280, totalling 738 million parameters. 

ZymCTRL is a decoder-only transformer model pre-trained on the BRENDA database (version July 2022). The pre-training was done on the raw sequences without FASTA headers, with the EC classes prepended to each sequence. The databases can be found here: xx.

ZymCTRL was trained with an autoregressive objective, i.e., the model learns to predict the next token given a sequence context. Because the first tokens on each sequence encode the EC numbers, the model learns the dependencies among EC classes and their corresponding sequences, and is able to _speak_ the enzyme language.
 
### **How to use ZymCTRL**
ZymCTRL can be used with the HuggingFace transformer python package. Detailed installation instructions can be found here: https://huggingface.co/docs/transformers/installation

Since ZymCTRL has been trained on the classical language model objective on enzyme sequences with their EC annotation, it particularly excels at generating enzyme sequences given a user-defined EC class, such as '1.1.1.2'. Besides, it can also be fine-tuned on a specific catalytic reaction by providing more sequences for a given EC class, such as sequences obtained with ancestral reconstruction methods.

**Example 1: Generating glucose oxidases (EC 1.1.3.4)**  
 
In the example below, ZymCTRL generates sequences that catalyze the reaction encoded by the EC number 1.1.3.4. Any other EC class can also be prompted instead. The model will generate the most probable sequences that follow the input.

```
from transformers import GPT2LMHeadModel, AutoTokenizer

enzyme_class = 1.1.3.4
device = torch.device("cuda") # if a GPU is available
tokenizer = AutoTokenizer.from_pretrained('/path/to/tokenizer')
model = GPT2LMHeadModel.from_pretrained('/path/to/output').to(device)
input_ids = tokenizer.encode(enzyme_class,return_tensors='pt').to(device)
# change max_length or num_return_sequences to your requirements
output = model.generate(input_ids, top_k=8, repetition_penalty=1.2, max_length=1024,
                        eos_token_id=1,pad_token_id=0,do_sample=True, num_return_sequences=100)
```

**Example 2: Finetuning on a set of user-defined sequences**  

This alternative option to the zero-shot generation permits further improve the model's confidence for EC number with few members. User-defined training and validation files containing the sequences of interest are provided to the model. After a short update of the model's weights, ZymCTRL will generate sequences that follow the input properties. This might not be necessary in cases where the model has already seen many sequences per EC class.

To create the validation and training file, it is necessary to (1) remove the FASTA headers for each sequence, (2) prepare the sequences in the format: EC number<sep><start>S E Q U E N C E<end><|endoftext|> and (3) split the originating dataset into training and validation files (this is often done with the ratio 90/10, 80/20 or 95/5). Then, to finetune the model to the input sequences, we can use the example below. Here we show a learning rate of 1e-06, but ideally, the learning rate should be optimised in separate runs. After training, the finetuned model will be stored in the ./output folder. Lastly, ZymCTRL can generate the tailored sequences as shown in Example 1:

```
python run_clm.py --model_name_or_path nferruz/ZymCTRL --train_file training.txt --validation_file validation.txt --tokenizer_name nferruz/ZymCTRL
 --do_train --do_eval --output_dir output --learning_rate 1e-06 

```
The HuggingFace script run_clm.py can be found here: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py

### **How to select the best sequences**

First of all, we recommend selecting only sequences where the padding token has been emitted. Because the generation occurs with a max_length parameter, Hugging Face generation function will truncate sequences that surpassed that length. Once the sequence has been emitted, select those with at least one <pad> token at the end. Otherwise you might be seeing truncated sequences by the length limit.

Besides, we've observed that perplexity values correlate with AlphaFold2's plddt. 
We recommend computing perplexity for each sequence as follows:

```
def calculatePerplexity(sequence, model, tokenizer):
    with torch.no_grad():
        outputs = model(sequence, labels=input_ids)
   loss, logits = outputs[:2]
    return math.exp(loss)
    
# Generate sequences by loading model and tokenizer (previously downloaded)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('/path/to/tokenizer') # replace with the actual path
model = GPT2LMHeadModel.from_pretrained('/path/to/output').to(device) 
output = model.generate("1.1.1.1", max_length=400, do_sample=True, top_k=8, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)

# Take (for example) the first sequence
sequence = output[0]
ppl = calculatePerplexity(sequence, model, tokenizer)
```

Where `ppl` is a value with the perplexity for that sequence.
We do not yet have a threshold as of what perplexity value gives a 'good' or 'bad' sequence, but given the fast inference times, the best is to sample many sequences, order them by perplexity, and select those with the lower values (the lower the better).


### **Training specs**
The model was trained on 48 NVIDIA A100 GPUs for 8 epochs, using a block size of 1024, and a total batch size of 768. The optimizer used was Adam (beta1 = 0.9, beta2 = 0.999) with a learning rate of 0.8e-04.