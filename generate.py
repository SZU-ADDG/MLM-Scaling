import torch
import os
import argparse
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
import datasets
from rdkit import Chem
from utils.chem_utils import reconstruct
from tqdm import tqdm


def Test(model, tokenizer, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation, device,
         output_file_path):
    complete_answer_list = []
    valid_answer_list = []
    model.eval()
    for x in tqdm(range(1000)):
        # place data on the correct device
        x = torch.tensor([1], dtype=torch.int64).unsqueeze(0)
        x = x.to(device)
        # pbar.set_description(f"iter {it}")
        with torch.no_grad():
            res_y = model.generate(x, tokenizer, max_new_tokens=max_seq_len,
                                   temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache,
                                   is_simulation=is_simulation)
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            complete_answer = f"{tokenizer.decode(x[0])}"

            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                complete_answer += answer[history_idx:]

                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            complete_answer = complete_answer.replace(" ", "").replace("[BOS]", "").replace("[EOS]", "")
            frag_list = complete_answer.replace(" ", "").split('[SEP]')
            try:
                frag_mol = [Chem.MolFromSmiles(s) for s in frag_list]
                mol = reconstruct(frag_mol)[0]
                if mol:
                    generate_smiles = Chem.MolToSmiles(mol)
                    valid_answer_list.append(generate_smiles)
                    answer = frag_list
                else:
                    answer = frag_list
            except:
                answer = frag_list
            complete_answer_list.append(answer)

    print(
        f"valid ratio:{len(valid_answer_list)}/{len(complete_answer_list)}={len(valid_answer_list) / len(complete_answer_list)}")
    os.makedirs(output_file_path, exist_ok=True)
    with open(os.path.join(output_file_path, f'small_complete_answer'), "w") as w:
        for j in complete_answer_list:
            if not isinstance(j, str):
                j = str(j)
            w.write(j)
            w.write("\n")
    w.close()
    with open(os.path.join(output_file_path, f'small_valid_answer'), "w") as w:
        for j in valid_answer_list:
            w.write(j)
            w.write("\n")
    w.close()


def main_test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f'cuda:{0}')

    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=4, n_head=4, n_embd=160)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(args.model_path, weights_only=True)  # or ./weights/fragpt_dpo.pt
    model.load_state_dict(checkpoint)
    start_time = time.time()
    Test(model, tokenizer, max_seq_len=1024, temperature=1.0, top_k=None, stream=False, rp=1., kv_cache=True,
         is_simulation=True, device=device, output_file_path=args.output_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"running time: {elapsed_time:.4f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--output_file_path', default='./output/qihua_SMILES-5M-100M-Epoch1-generation')
    parser.add_argument('--model_path', default='./weights/qihua_SMILES-5M-100M-Epoch1.pt', help='model weight file path')

    opt = parser.parse_args()

    main_test(opt)