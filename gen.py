import os, glob, re
import gpt_2_simple as gpt2
import tensorflow as tf

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
# model_dir = 'models/355M/'

# book_path = "data/born_a_crime/test-utf8.txt"
book_path = "data/prefix.txt"
with open(book_path, encoding='utf8') as fin:
    # fin.seek(0)
    prefix = fin.read()

# run_name = 'page'

# models = glob.glob("checkpoint/run_b2_2_noMem_s15/*.meta")

# cp_path = 'models/355M'

for i in range(50, 550, 10):

    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()

    cp_path = 'checkpoint/train_longer'
    cp = i
    with open('checkpoint/train_longer/checkpoint', 'w') as f:
        f.write(f"model_checkpoint_path: \"model-{cp}\"\nall_model_checkpoint_paths: \"model-{cp}\"")

    gen_path = 'born_again_a_crime7.txt'
    f=open(gen_path, "a+", encoding='utf8')
    gpt2.load_gpt2(sess,run_name="", checkpoint_dir=cp_path)
    old_prefix = ' '.join(prefix.split()[-63:])
    prefix = gpt2.generate(sess, run_name="", checkpoint_dir=cp_path, length=31, temperature=0.7,top_k=50,top_p=0, sample_delim=' ', prefix=old_prefix, include_prefix=True, truncate='<|endoftext|>', return_as_list=True)[0]
    f.write(prefix.split(old_prefix)[-1])

    sess.close()
    # wait = input("PRESS ENTER TO CONTINUE.")


