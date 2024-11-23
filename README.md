Requirements
Ensure you have the following installed on your system:

Python 3.8+ or (3.11.8 as used in this project)
Git
A virtual environment manager (breakout_env)
Required libraries (listed in requirements.txt)

Setup

1. Clone the Repository:
   (https://github.com/RuthBiney/Deep_Q_leaarning.git)

2. Set Up a Virtual Environment:
   python -m venv breakout_env
   source venv/bin/activate # For Linux/Mac
   venv\Scripts\activate # For Windows

3. Activate the breakout_env
   breakout_env\Scripts\activate

4. Install Dependencies:
   pip install -r requirements.txt

5. To run install the following:

pip install gym stable-baselines3
pip install "gym[atari]"
pip install 'shimmy>=2.0

6. Run the Training Script:
   python train.py

7. To run the Game Script:
   python play.py

Video link:
https://drive.google.com/file/d/1wRA8KT61RqzsvofoyeeFZB6i6q__NlYN/view?usp=sharing

The file for the policy.h5 and models were veryheavy and since could not push it to my github but I downloaded in to my drive
link to the policy.h5 and models: https://drive.google.com/drive/folders/1qAA6wCmwGJQZky1AoDiyULzHLyw7ieYK?usp=sharing
