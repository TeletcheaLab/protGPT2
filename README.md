# **ProtGPT2**

ProtGPT2 ([paper](https://www.biorxiv.org/content/10.1101/2022.03.09.483666v1) is a language model that speaks the protein language and can be used for de novo protein design and engineering.
ProtGPT2 is based on the GPT2 Transformer architecture and contains 36 layers with a model dimensionality of 1280, totalling 738 million parameters. 


## **Model description**
ProtGPT2 is a decoder-only transformer model pre-trained on the protein space, database UniRef50 (version 2021_04). The pre-training was done on the raw sequences without FASTA headers, details of training and datasets can be found here: https://huggingface.co/datasets/nferruz/UR50_2021_04

ProtGPT2 was trained in a self-supervised fashion, this means that the raw sequence data was using during training, without including the annotation of sequences. In particular, ProtGPT2 was trained using an causal modelling objective, in which the model is trained to predict the next token (or in this case, oligomer) in the sequence.
 By doing so the model learns an internal representation of the 'protein language', in other words, learns the protein language and is able to <em>speak</em> it.
 
ProtGPT2 generates sequences that populate unseen regions of the protein space and which besides are globular, ordered and distant from current natural sequences. 

### **Intended uses and limitations**

Since ProtGPT2 has been trained on the classical language model objective, it excels at generating protein sequences. It can be used to generate sequences in a zero-shot fashion or to generate sequences of a particular type after finetuning on a user-defined dataset. 

Example 1: Generating de novo proteins in a zero-shot fashion. We recommend the following parameters:
```
>>> from transformers import pipeline
>>> protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
>>> sequences = protgpt2("M", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
>>> for seq in sequences:
        print(seq):
 {'generated_text': 'MINDLLDISRIISGKMTLDRAEVNLTAIARQVVEEQRQAAEAKSIQLLCSTPDTNHYVFG\nDFDRLKQTLWNLLSNAVKFTPSGGTVELELGYNAEGMEVYVKDSGIGIDPAFLPYVFDRF\nRQSDAADSRNYGGLGLGLAIVKHLLDLHEGNVSAQSEGFGKGATFTVLLPLKPLKRELAA\nVNRHTAVQQSAPLNDNLAGMKILIVEDRPDTNEMVSYILEEAGAIVETAESGAAALTSLK\nSYSPDLVLSDIGMPMMDGYEMIEYIREWKTTKGG'}
{'generated_text': 'MQGDSSISSSNRMFT\nLCKPLTVANETSTLSTTRNSKSNKRVSKQRVNLAESPERNAPSPASIKTNETEEFSTIKT\nTNNEVLGYEPNYVSYDFVPMEKCNLCNENCSIELASLNEETFVKKTICCHECRKKAIENA\nENNNTKGSAVSNNSVTSSSGRKKIIVSGSQILRNLDSLTSSKSNISTLLNPNHLAKLAKN\nGNLSSLSSLQSSASSISKSSSTSSTPTTSPKVSSPTNSPSSSPINSPTP'}
{'generated_text': 'M\nSTHVSLENTLASLQATFFSLEARHTALETQLLSTRTELAATKQELVRVQAEISRADAQAQ\nDLKAQILTLKEKADQAEVEAAAATQRAEESQAALEAQTAELAQLRLEKQAPQHVAEEGDP\nQPAAPTTQAQSPVTSAAAAASSAASAEPSKPELTFPAYTKRKPPTITHAPKAPTKVALNP\nSTLSTSGSGGGAKADPTPTTPVPSSSAGLIPKALRLPPPVTPAASGAKPAPSARSKLRGP\nDAPLSPSTQS'}
{'generated_text': 'MVLLSTGPLPILFLGPSLAELNQKYQVVSDTLLRFTNTV\nTFNTLKFLGSDS\n'}
{'generated_text': 'M\nNNDEQPFIMSTSGYAGNTTSSMNSTSDFNTNNKSNTWSNRFSNFIAYFSGVGWFIGAISV\nIFFIIYVIVFLSRKTKPSGQKQYSRTERNNRDVDSIKRANYYG\n'}
{'generated_text': 'M\nEAVYSFTITETGTGTVEVTPLDRTISGADIVYPPDTACVPLTVQPVINANGTWTLGSGCT\nGHFSVDTTGHVNCLTGGFGAAGVHTVIYTVETPYSGNSFAVIDVNVTEPSGPGDGGNGNG\nDRGDGPDNGGGNNPGPDPDPSTPPPPGDCSSPLPVVCSDRDCADFDTQAQVQIYLDRYGG\nTCDLDGNHDGTPCENLPNNSGGQSSDSGNGGGNPGTGSTHQVVTGDCLWNIASRNNGQGG\nQAWPALLAANNESITNP'}
{'generated_text': 'M\nGLTTSGGARGFCSLAVLQELVPRPELLFVIDRAFHSGKHAVDMQVVDQEGLGDGVATLLY\nAHQGLYTCLLQAEARLLGREWAAVPALEPNFMESPLIALPRQLLEGLEQNILSAYGSEWS\nQDVAEPQGDTPAALLATALGLHEPQQVAQRRRQLFEAAEAALQAIRASA\n'}
{'generated_text': 'M\nGAAGYTGSLILAALKQNPDIAVYALNRNDEKLKDVCGQYSNLKGQVCDLSNESQVEALLS\nGPRKTVVNLVGPYSFYGSRVLNACIEANCHYIDLTGEVYWIPQMIKQYHHKAVQSGARIV\nPAVGFDSTPAELGSFFAYQQCREKLKKAHLKIKAYTGQSGGASGGTILTMIQHGIENGKI\nLREIRSMANPREPQSDFKHYKEKTFQDGSASFWGVPFVMKGINTPVVQRSASLLKKLYQP\nFDYKQCFSFSTLLNSLFSYIFNAI'}
{'generated_text': 'M\nKFPSLLLDSYLLVFFIFCSLGLYFSPKEFLSKSYTLLTFFGSLLFIVLVAFPYQSAISAS\nKYYYFPFPIQFFDIGLAENKSNFVTSTTILIFCFILFKRQKYISLLLLTVVLIPIISKGN\nYLFIILILNLAVYFFLFKKLYKKGFCISLFLVFSCIFIFIVSKIMYSSGIEGIYKELIFT\nGDNDGRFLIIKSFLEYWKDNLFFGLGPSSVNLFSGAVSGSFHNTYFFIFFQSGILGAFIF\nLLPFVYFFISFFKDNSSFMKLF'}
{'generated_text': 'M\nRRAVGNADLGMEAARYEPSGAYQASEGDGAHGKPHSLPFVALERWQQLGPEERTLAEAVR\nAVLASGQYLLGEAVRRFETAVAAWLGVPFALGVASGTAALTLALRAYGVGPGDEVIVPAI\nTFIATSNAITAAGARPVLVDIDPSTWNMSVASLAARLTPKTKAILAVHLWGQPVDMHPLL\nDIAAQANLAVIEDCAQALGASIAGTKVGTFGDAAAFSFYPTKNMTTGEGGMLVTNARDLA\nQAARMLRSHGQDPPTAYMHSQVGFN'}
```

Example 2: Finetuning on a set of user-defined sequences. This example finetunes using a user-defined training and validation files that contain a set of sequences of interest. The create the validation and training file, it is necessary to (1) substitute the FASTA headers for each sequence with the tag "<|endoftag|>" and (2) split the originating dataset into training and validation files, (this is often done with the ratio 90/10, 80/20 or 95/5). Here we show a learning rate of 1e-06, but ideally should be optimized in separate runs. After training, the finetuned model will be stored in the ./output folder. This model can be used as in the example above to generate tailored sequences.

The HuggingFace script can be found here: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py

```
python run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file training.txt --validation_file validation.txt --tokenizer_name nferruz/ProtGPT2
 --do_train --do_eval --output_dir output --learning_rate 1e-06 

```

### **Training specs**
The model was trained on 128 NVIDIA A100 GPUs for 50 epochs, using a block size of 512, and a total batch size of 1024 (65,536 tokens per batch). The optimizer used was Adam (beta1 = 0.9, beta2 = 0.999) with a learning rate of 1e-3 . 