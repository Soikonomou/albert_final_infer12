from src.tasks.preprocessing_funcs import preprocess_fewrel
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained
import logging
from argparse import ArgumentParser
import string
from nltk.tokenize import TweetTokenizer
import nltk
import pandas as pd
import requests
import re
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
'''
This fine-tunes the BERT model on SemEval task
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


def ner_relation(text):
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel')
    parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT',
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT',
                        help="test data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0,
                        help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1,
                        help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    # mixed precision doesn't seem to train well
    parser.add_argument("--fp16", type=int, default=0,
                        help="1: use mixed precision ; 0: use floating point 32")
    parser.add_argument("--num_epochs", type=int, default=20, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--model_no", type=int, default=1, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT''')
    parser.add_argument("--train", type=int, default=0, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")

    args = parser.parse_args()

    if args.train == 1:
        net = train_and_fit(args)

    if args.infer == 1:
        inferer = infer_from_trained(args, detect_entities=False)
        body = text
        data = {
            "_id": "abc",
            "body": body,
            "sentences": {
                "0": {
                    "text": body,
                    "offset": "0"
                }
            }
        }
        result = requests.post(url="http://10.142.0.204:4020/entity_extraction", json=data)
        output = result.json()
        n_entities = []
        document = ' '.join(output['body'].split())
        quotes = re.findall(r'".*?"', document)
        for i in range(0, len(quotes)):
            document = document.replace(quotes[i], ''.join(nltk.sent_tokenize(quotes[i])))
        phrases = nltk.sent_tokenize(document)
        X_sent = [phrase for phrase in phrases if len(nltk.word_tokenize(
            phrase.translate(str.maketrans('', '', string.punctuation)))) > 5]
        X_test1 = [article.translate(str.maketrans('', '', string.punctuation))
                   for article in X_sent]
        X_test2 = [' '.join(tknzr.tokenize(a)) for a in X_test1]
        X_test3 = [re.sub('\s+', ' ', s) for s in X_test2]
        X_test4 = [re.sub('[^A-Za-z0-9]+', ' ', s) for s in X_test3]
        X_test5 = [re.sub(' +', ' ', s) for s in X_test4]
        X_test6 = [re.sub(r'http\S+', ' ', s) for s in X_test5]
        X_test = [a.strip() for a in X_test6]
        for i in range(0, len(output['sentences']['0']['entities_e2e'])):
            n_entities.append(output['sentences']['0']['entities_e2e'][i]['URI'])
        for entities in n_entities:
            if 'Mr' in entities:
                entities.replace('Mr', '').strip()
            elif 'Mrs' in entities:
                entities.replace('Mrs', '').strip()
        results = []
        for sent in X_test:
            pairs = [x in sent for x in n_entities]
            entities = []
            for i in range(0, len(pairs)):
                if pairs[i] == True:
                    entities.append(n_entities[i])
            words = nltk.word_tokenize(sent)
            order = []
            for i in range(0, len(entities)):
                for j in range(0, len(words)):
                    if entities[i] == words[j]:
                        order.append(i)
                        break
            ordered_entities = [x for _, x in sorted(zip(order, entities))]
            for previous, current in zip(ordered_entities, ordered_entities[1:]):
                sent_new = re.sub(previous, '[E1]{}[/E1]'.format(previous), sent)
                sent_new = re.sub(current, '[E2]{}[/E2]'.format(current), sent_new)
                result = inferer.infer_sentence(sent_new, detect_entities=False)
                results.append([previous, current, result])
    return(results)
