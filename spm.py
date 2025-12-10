import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=data/train.txt --model_prefix=tokenizer/spm --vocab_size=37000 --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3')