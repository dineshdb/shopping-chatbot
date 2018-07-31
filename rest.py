from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
import nltk
from classifier.DNN_multilabel import eval1
from classifier.CNN_binary import eval2
from flask_cors import CORS
from rivescript import RiveScript
from rscript import rive

app = Flask(__name__)
CORS(app)

@app.route("/reply/<name>")
def gets(name):
	x_raw = [name] #input sentence
	x_test = eval2.np.array(list(eval2.vocab_processor.transform(x_raw)))

	# Generate batches for one epoch
	batches = eval2.processData.batch_iter(list(x_test), 1, 1, shuffle=False)


	for x_test_batch in batches:
		batch_predictions = eval2.sess.run(eval2.predictions, {eval2.input_x: x_test_batch, eval2.dropout_keep_prob: 1.0})
		if(batch_predictions==[1]):
                    if(eval1.categories[eval1.np.argmax(eval1.model.predict([eval1.get_tf_record(name)]))]=='checkout'):
                             reply = rive.bot.reply("localuser", 'checkout') 
                             return reply
                    elif(eval1.categories[eval1.np.argmax(eval1.model.predict([eval1.get_tf_record(name)]))]=='cart'):
                         reply = rive.bot.reply("localuser", 'cart') 
                         return reply    
                    elif(eval1.categories[eval1.np.argmax(eval1.model.predict([eval1.get_tf_record(name)]))]=='explore_items'):
                         nouns = ['hi']
                         for word,pos in nltk.pos_tag(nltk.word_tokenize(str(name))):
                             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                                 nouns.append(word)
                                 
                         for noun in nouns:
                             items = ['laptop','iphone','shirt']
                             if noun in items:
                                 reply = rive.bot.reply("localuser", 'explore '+noun) 
                                 return reply
                         else:
                            return 'Please ensure the item you are searching is available'
                    elif(eval1.categories[eval1.np.argmax(eval1.model.predict([eval1.get_tf_record(name)]))]=='show categories'):
                         reply = rive.bot.reply("localuser", 'show categories') 
                         return reply   
			 
			 
		else:
                    reply = rive.bot.reply("localuser", name) 
                    return reply                  
                     
         
@app.route("/entityExtract/<name>")
def get(name):
    nouns = []
    for word,pos in nltk.pos_tag(nltk.word_tokenize(str(name))):
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                 nouns.append(word)
    return jsonify(nouns)
	
if __name__ == '__main__':
     app.run(port='5000')
     