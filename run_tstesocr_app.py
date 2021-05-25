import time, json, os, datetime
from PIL import Image
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import lib.utils as utils 
import base64 
from lib.utils import Base64toimg    # 解碼 base64 
import re 
# import base64
# import datetime
import hashlib

# import logging
# log = logging.getLogger('werkzeug')
# log.disabled = True

# 定義參數

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'rchiu3210@gmail.com'   #
SALT = 'dwkljgc;h45137~974#%#@$&4@#%'   #
#########################################

def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def prepa_str_list():
    dict_path = '.\\dlocr\\dictionary\\'
    dict_file = 'char_std_ES0000_800.txt'
    words = ''
    f = open(dict_path+dict_file, 'r+', encoding='utf-8')
    words = f.read()
    words = words.replace('blank\n','')  
    words = words.replace(' \n','') 
    words = words.replace('\n','')
    f.close()
    str_list = []
    for w in words:
        str_list.append(w) 
    return str_list

str_list = prepa_str_list()

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
# print('os.environ["CUDA_VISIBLE_DEVICES"]=', os.environ["CUDA_VISIBLE_DEVICES"]) 
 
import lib.fun_model_ocr as model_ocr  

default_api_config_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "./config/api-default.json")
# default_api_config_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "./config/api-default-t.json")
def load_config(api_config_path):
    with open(api_config_path, "r") as infile:
        return dict(json.load(infile))
global _api_config
_api_config = load_config(api_config_path=default_api_config_path)
ocrpath =_api_config['keyfile']
f = open (ocrpath, 'r')
ilist = f.read().split("\n")
f.close()
ocr=''
for fif in ilist:
    if ocr == '':
        ocr = fif.replace('OCR=','')

app = Flask(__name__)
api = Api(app)

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        # posted_data = request.data
        # docname = posted_data['docname']
        esun_uuid = posted_data['esun_uuid']
        esun_timestamp = posted_data['esun_timestamp']
        imagestr = posted_data['image']
        retry = posted_data['retry']
        origarr, img64arr = Base64toimg.strbase64to64array(imagestr)
        # imagestr = base64.decodebytes(imagestr.encode('utf-8'))
        # origarr = imagestr
        # mddia = posted_data['media']

        # import lib.fun_idclassfication_txt as model_idclf
        # imagestr = model_idclf.decrypto_imagestr(imagestr, ocr)
        # origarr, img64arr = model_idclf.proc_imagestr_64img(imagestr)
        # label, outimg  = model_idclf.predict(origarr, img64arr)
        
        t = datetime.datetime.now()
        ts = str(int(t.utcnow().timestamp())) 
        # print(t, ts)   
        server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)
        # print(server_uuid)

        try:
            texts, cost = model_ocr.predict_ix0090ocr(origarr)
            print('result:{}.'.format(texts))
            texts = re.sub("[a-zA-Z0-9() ]",'',texts)
            logstr = '{},{},{},{},{},{},{}\n'.format(int(time.time()), server_uuid, texts, esun_uuid, esun_timestamp, retry, imagestr)
            # print(logstr)
            f=open('./work/output/log.txt', 'a+', encoding='utf-8')
            f.write(logstr)
            f.close() 
            print('write logs into logfile ok.')

            if len(texts) == 0:
                out_res={"esun_uuid": esun_uuid,
                        "server_uuid": server_uuid,
                        "server_timestamp": int(time.time()),
                        "answer": 'isnull'}
                print(out_res)
            elif len(texts) > 1:
                if texts[0] in str_list:
                    out_res={"esun_uuid": esun_uuid,
                        "server_uuid": server_uuid,
                        "server_timestamp": int(time.time()),
                        "answer": texts[0]}
                    print(out_res)
                else:
                    out_res={"esun_uuid": esun_uuid,
                        "server_uuid": server_uuid,
                        "server_timestamp": int(time.time()),
                        "answer": 'isnull'}
                    print(out_res)
            else: # len(texts) == 1
                if texts[0] in str_list:
                    out_res={"esun_uuid": esun_uuid,
                        "server_uuid": server_uuid,
                        "server_timestamp": int(time.time()),
                        "answer": texts[0]}
                    print(out_res)
                else:
                    out_res={"esun_uuid": esun_uuid,
                        "server_uuid": server_uuid,
                        "server_timestamp": int(time.time()),
                        "answer": 'isnull'} 
                    print(out_res)               
        except:                
            out_res={"esun_uuid": esun_uuid,
                        "server_uuid": server_uuid,
                        "server_timestamp": time.time(),
                        "answer": 'isnull',
                        "errcode": 'err-0000'}
            print(out_res)
        return jsonify(out_res)

        # try:
        #     if label[0] == '00':
        #         out_res=[docname, '9999', '黑白證件不須辨識', 0]
        #     else:
        #         texts, cost = model_ocr.predict_ix0090ocr(outimg)  # 有排序 sb_default
        #         texts = utils.Ocr_pred_fxstr.fxstr_default(texts)
        #         out_res=[docname, '0000', docname.split('_')[1][:4], texts, cost]   
        # except:                
        #     out_res=[docname, '9999', 'unknow err', 0]
        # return jsonify({"texts": out_res})
api.add_resource(MakePrediction, '/inference')

if __name__ == '__main__':      
    app.run(debug=False, threaded=False, host='192.168.0.30', port=5000)  # , host='192.168.8.101' # '100.68.189.170' 
