# Torch Text Summarization 

This project is designed to offer an introduction to the workings of transformers, providing an intuitive understanding of their mechanisms. It centers around the task of dialogue summarization. Utilizing a **sequence-to-sequence** (seq-to-seq) model, which incorporates both an encoder and a decoder, facilitates this process. To accelerate convergence during the training phase, the teacher forcing strategy is employed. The dataset employed for training can be found at 'data/corpus'.


## Results

It's important to note that achieving optimal results with Transformers typically requires substantial data and computational resources. Given the limited scope of this project, the summarizations may not be of the highest quality, but they are sufficient to demonstrate that valuable learning is taking place.

| Dialogue    | Summary(Pred)|Summary (True)| Split     |
| ----------- | -----------|-------------- |----------|
| Gunther: did you pay for coffee?</br>Chandler: uhh.. i guess not xD but it's okay i'll pay him tomorrow</br>Gunther: -_- | chandler will pay for the coffee tomorrow | Chandler will pay for his coffee tomorrow. | Train |
|Silvia: can you collect me from the party tonight</br>Lonyo: ok</br>Lonyo: what time?</br>Silvia: i dont know yet</br>Silvia: can i let you know in the night?</br>Lonyo: ok</br>Lonyo: i will wait for your call | ilvia will let her know if she can come tonight | Silvia will let Lonyo know what time to pick her up from the party tonight. | Train |
| Jimmy: Can I borrow your car?</br>Max: No, Jimmy.</br>Max: You have a bike! |jimmy will borrow his own| Jimmy asked Max if he could borrow his car but Max refused. | Train |
|Hannah: Hey, do you have Betty's number?</br>Amanda: Lemme check</br>Hannah: <file_gif></br>Amanda: Sorry, can't find it.</br>Amanda: Ask Larry</br>Amanda: He called her last time we were at the park together</br>Hannah: I don't know him well</br>Hannah: <file_gif></br>Amanda: Don't be shy, he's very nice</br>Hannah: If you say so..</br>Hannah: I'd rather you texted him</br>Amanda: Just text him 🙂</br>Hannah: Urgh.. Alright</br>Hannah: Bye</br>Amanda: Bye bye|hannah is angry because of the time|Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.|Test |
| Mike: Do u have new John's number?</br>Ann: No, u should ask Mary.</br>Mike: Ok, thank u :*|mike is going to ask mike|Mike will ask Mary for John's new number.|Test|
|Kim: What kind of gift would you like to get?</br>Kim: Mom's asking.</br>Harry: Haha. No need for a gift for me :D</br>Harry: But you can tell your mom I just bought a new sofa and I need pillows.</br>Harry: If she asks for the colour, tell her that grey is the best :D</br>Kim: Sure! Thanks for info :)|mike is going to ask mike|Kim is about to tell mom that Harry bought a new sofa, and he needs grey pillows.|Test |

## Installation

Before starting make sure you have python 3.9+ installed.

1. `git clone https://github.com/abtraore/torch-text-summarization.git`
2. `python -m venv env`
3. `pip install -r requirement.txt`


## Usage

