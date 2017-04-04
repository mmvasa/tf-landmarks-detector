
#!/bin/bash

rm -r videos/*.png
ffmpeg -i $1 -vf fps=2 videos/out%d.png
python3 extract_face.py
python3 model_eval.py
