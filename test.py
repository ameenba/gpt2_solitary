import os, glob, re
import gpt_2_simple as gpt2
import tensorflow as tf
from timeit import default_timer as timer

# def merge_files(merged_file, directory):

    # read_files = glob.glob(directory + "*.xml")
    # write_file = directory + merged_file + ".txt"

    # with open(write_file, "wb") as outfile:
    #     for f in read_files:
    #         with open(f, "rb") as infile:
    #             outfile.write(infile.read())

    # return write_file

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

iters = 50
train_path = "data/born_a_crime/trevor-noah.txt"
run_name = 'train_longer'

with open(train_path, encoding='utf8') as fin:
    # fin.seek(0)
    prefix = fin.read()


steps = 50

for i in range(iters):

    #not needed gpt2 can take a directory
    # merged_file = merge_files(directory.split('/')[-2], directory)
    tf.reset_default_graph()
    
    sess = gpt2.start_tf_sess()

    gpt2.finetune(sess, train_path, model_name=model_name, overwrite=False, steps=steps, batch_size=2, run_name=run_name)   # steps is max number of training steps

    gen_path = 'gen/' + run_name + '/result_' + str(i) + '.txt'
    f=open(gen_path, "a+", encoding='utf8')

    for j in range(iters):
        prefix = ' '.join(prefix.split()[-35:])
        print(prefix)
        prefix = gpt2.generate(sess, temperature=1.15, top_k=30, sample_delim=' ', run_name=run_name, prefix=prefix, include_prefix=True, truncate='<|endoftext|>', return_as_list=True)[0]
        prefix = re.sub(r' {2,}' , ' ', prefix)
        prefix = re.sub(r'\t{2,}' , '\t', prefix)
        prefix = re.sub(r'\n{2,}' , '\n', prefix)
        f.write(' '.join(prefix.split()[35:]))
        
    f.close()
        
    train_path = gen_path
    steps = 10
    sess.close()

