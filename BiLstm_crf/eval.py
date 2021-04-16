import util as util
import os
import re
import torch
import BiLSTM_CRF
import codecs

EPOCHES = 300

EMBEDDING_DIM = 300
# 隐层向量大小
HIDDEN_DIM = 300
# 批量大小
BATCH_SIZE = 128
# 学习率
lr = 0.0001

TRAIN_DATA_PATH = r"../data/NER_corpus_chinese-master/Peoples_Daily/rmrb4train.csv"
MODEL_PATH = os.path.join(r'model/', "model_emb_{}_hidden_{}_batch_{}_baseline2".format(EMBEDDING_DIM, HIDDEN_DIM,BATCH_SIZE))
NEWS_PATH = r"../newsin"


_, word_to_ix, tag_to_ix = util.data_prepare(TRAIN_DATA_PATH)
OrderedDict = torch.load(MODEL_PATH)
len_ = len(OrderedDict["word_embeds.weight"])

model = BiLSTM_CRF(len_, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

ix_to_tag = {v: k for k, v in tag_to_ix.items()}


for parent, dirnames, filenames in os.walk(NEWS_PATH):
    for filename in filenames:
        file_path_in = os.path.join(parent, filename)
        print("transforming.....", file_path_in)
        file_path_out = os.path.join(parent, r"..\\NERout\\ner_{}".format(filename))
        test_d = codecs.open(file_path_in, 'r', 'utf-8').read()
        test_d = strQ2B(test_d)
        test_d = re.sub(r"\n\n","",test_d)
        test_d_Lst = cut_sentence(test_d)
        result = set()

        for sent in test_d_Lst:
            sent = cleanSent(sent)
            if len(sent) < 4: continue
            test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in sent]
            test_res_ = torch.tensor(test_id, dtype=torch.long)
            eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]

            tag2word_res = util.tag2word(sent,eval_res)
            for s in tag2word_res:
                result.add(s)

        with codecs.open(file_path_out, 'w+', 'utf-8') as surveyp:
            surveyp.write(",\n".join(result))



# test_sent = '中国邮政10日在拉萨发行《川藏青藏公路建成通车六十五周年》纪念邮票1套2枚'
# test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in test_sent]
# test_res_ = torch.tensor(test_id,dtype=torch.long)
# eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]
# print(eval_res)
# # #['O', 'O', 'O', 'O', 'B_PERSON', 'E_PERSON', 'O', 'B_PERSON', 'E_PERSON', 'O', 'O', 'B_LOCATION', 'E_LOCATION', 'O', 'B_LOCATION', 'M_LOCATION', 'M_LOCATION', 'E_LOCATION', 'O', 'O', 'O', 'O', 'O', 'O']