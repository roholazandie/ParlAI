#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tornado.websocket import WebSocketHandler
import uuid
import logging


def get_rand_id():
    return str(uuid.uuid4())


class MessageSocketHandler(WebSocketHandler):
    subs = {}

    def __init__(self, *args, **kwargs):
        def _default_callback(message, socketID):
            logging.warn(f"No callback defined for new WebSocket messages.")

        self.message_callback = kwargs.pop('message_callback', _default_callback)
        self.sid = get_rand_id()
        super().__init__(*args, **kwargs)

    def open(self):
        """
        Opens a websocket and assigns a random UUID that is stored in the class-level
        `subs` variable
        """
        if self.sid not in self.subs.values():
            self.subs[self.sid] = self
            logging.info(f"Opened new socket from ip: {self.request.remote_ip}")
            logging.info(f"Current subscribers: {self.subs}")

    def on_close(self):
        """Runs when a socket is closed"""
        del self.subs[self.sid]

    def on_message(self, message):
        """Callback that runs when a new message is received from a client"""
        logging.info('websocket message from client: {}'.format(message))
        self.message_callback(message, self.sid)

    def check_origin(self, origin):
        return True
