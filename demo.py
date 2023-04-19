# -*- coding: utf-8 -*-

import json
import sys

import requests
import urllib3

from config.msaz import API_URL, API_KEY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Chat:
    def __init__(self):
        self.session = []
        self.chat_q = {'role': 'user', 'content': None}
        self.chat_a = {'role': 'assistant', 'content': None}

    def _concat_session(self, q, a):
        chat_q = self.chat_q.copy()
        chat_a = self.chat_a.copy()
        chat_q['content'] = q
        chat_a['content'] = a
        self.session.append(chat_q)
        self.session.append(chat_a)

    def send_q(self, q):
        print('Thinking...', flush=True)
        chat_q = self.chat_q.copy()
        chat_q['content'] = q

        proxies = {
            'http': 'http://{uname}:{pw}@some.proxy.com:{port}',
            'https': 'http://{uname}:{pw}@some.proxy.com:{port}'
        }
        url = API_URL
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'api-key': API_KEY,
            'Content-Type': 'application/json'
        }
        params = {'api-version': '2023-03-15-preview'}
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': self.session + [chat_q]
        }
        res = requests.post(url, headers=headers, params=params, data=json.dumps(data), proxies=None)
        if not res.ok:
            raise Exception(f'Error: {res.text}')

        a = res.json()['choices'][0]['message']['content']
        return a

    def chat(self):
        while True:
            print('Q: \n')
            q = read_block()
            a = self.send_q(q)
            self._concat_session(q, a)
            print('A: \n' + a)
            if q == 'bye':
                break


def read_block():
    blk = ''
    for line in sys.stdin:
        blk += line
        if blk.endswith('\n\n\n'):
            return blk.strip()


if __name__ == '__main__':
    c = Chat()
    c.chat()
