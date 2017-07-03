#This script is independet of lib or python version (tested on python 2.7 and 3.5)

try:
	import telegram
except ImportError:
	pass

import json

try:
	import urllib.request as req
except ImportError:
	import urllib as req

import urllib
from pprint import pprint 

#token that can be generated talking with @BotFather on telegram
my_token = '363083153:AAGFIcfkGdxRmEqYwSrq4h91t83mWMfgNgc'

#chat that will recieves the messages from bot
default_chat_id = 169036135

def send(msg, token=my_token, chat_id=default_chat_id):
	"""
	Send a mensage to a telegram user specified on chatId
	"""
	try:
		bot = telegram.Bot(token=token)
		bot.sendMessage(chat_id=chat_id, text=msg, parse_mode=telegram.ParseMode.HTML)
	except NameError:
	#case oesn't have telegram lib installed
		sendViaUrl(msg,token,chat_id)

def getChats(token=my_token):
	"""
	Function to get the id of the users that have talked with this bot
	"""
	result = req.urlopen("https://api.telegram.org/bot" + token + "/getUpdates").read()
	messages = json.loads(result.decode('utf-8'))['result']
	#print(result)
	talks = {}
	for data in messages:
		#pprint(data)
		chat = data['message']['chat']
		talks[chat['id']] = "{} {} (@{})".format(chat['first_name'],chat['last_name'],chat['username'])
	pprint(talks)

def sendViaUrl(msg, token=my_token, chat_id=default_chat_id):
	
	data = { "chat_id": chat_id, "text": msg }
	try:
		sendData = urllib.parse.urlencode(data).encode('ascii')
	except Exception:
    #caso of python version 2.7
		import urllib
		sendData = urllib.urlencode(data)

	# Send a message to a chat room (chat room ID retrieved from getUpdates)
	result = req.urlopen("https://api.telegram.org/bot" + token + "/sendMessage", sendData).read()

def setChatId(id):
	"""
	Set a default chat to your bot
	"""
	default_chat_id = id