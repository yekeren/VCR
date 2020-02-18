#!/bin/sh

ps x | grep python | grep -v grep | grep -v tensorboard | awk '{print $1}' | xargs kill
