#!/bin/bash
echo -n "enter prompt (e.g. a bento box): "
while read -r PROMPT && [ -z "$PROMPT" ]; do echo -n "enter prompt (e.g. a bento box): "; done
echo -n "enter negative prompt (e.g. low-res,blurry): "
while read -r N_PROMPT && [ -z "$N_PROMPT" ]; do echo -n "enter negative prompt (e.g. low-res,blurry): "; done

if [[ -n "$PROMPT" ]];
then
    curl -X POST http://127.0.0.1:3000/txt2img -H 'Content-Type: application/json' -d "{\"prompt\":\"$PROMPT\",\"negative_prompt\":\"$N_PROMPT\"}" --output output.jpg
else
    echo "No input"
fi
