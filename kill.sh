#!/bin/sh

ps x | grep python | grep -v grep | awk '{print $1}' | xargs kill
