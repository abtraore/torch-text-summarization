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

To set up the environment for the Torch Text Summarization tool, follow these steps. This guide assumes you have Python 3.9 or newer installed on your system. If you're unsure about your Python version, you can check it by running `python --version` in your terminal or command prompt.

### Step 1: Clone the Repository

First, you need to clone the repository to your local machine. Open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/abtraore/torch-text-summarization.git
```

This command downloads the codebase of the Torch Text Summarization project to your current directory.

### Step 2: Create a Virtual Environment

After cloning the repository, navigate into the project directory:

```bash
cd torch-text-summarization
```

Then, create a virtual environment named `env` within the project directory. This environment will contain all the necessary Python packages and will isolate them from the rest of your system.

```bash
python -m venv env
```

### Step 3: Activate the Virtual Environment

Before installing the dependencies, you need to activate the virtual environment. The activation command differs depending on your operating system.

- On Windows, run:

  ```bash
  env\Scripts\activate
  ```

- On Unix or MacOS, run:

  ```bash
  source env/bin/activate
  ```

### Step 4: Install Dependencies

With the virtual environment activated, install the project's dependencies using pip:

```bash
pip install -r requirements.txt
```

This command reads the `requirements.txt` file in the project directory and installs all the listed packages.



## Usage

This section guides you through the process of using the command-line interface to perform text summarization. Ensure you have the necessary environment setup before proceeding.

When documenting configurations or parameters in GitHub READMEs or similar documentation, using tables is an excellent way to present information in an organized and accessible manner. Here's a refined version of your configuration documentation that aligns with common practices:

### Configurations

The model's configuration settings are defined within `utils/config.py`. This file contains a class named `TransformerConfig`, which encapsulates various hyperparameters relevant to the model's architecture and training process. Below is a table summarizing these hyperparameters for quick reference:

| Hyper-parameter        | Description                                                   |
|------------------------|---------------------------------------------------------------|
| `n_blocks`             | Number of blocks in the encoder and decoder                   |
| `n_heads`              | Number of attention heads in multi-head attention             |
| `d_model`              | Size of the embeddings                                        |
| `fully_connected_dim`  | Dimension of the feed-forward network (FFN)                   |
| `max_position_encoding`| Maximum position encoding for the positional encoding         |
| `max_seq_length_input` | Maximum sequence length for the input                         |
| `max_seq_length_target`| Maximum sequence length for the target/output                 |
| `dropout_rate`         | Dropout rate applied in the feed-forward network              |
| `epochs`               | Number of epochs for training                                 |
| `batch_size`           | Batch size during training                                    |
| `weights_path`         | Directory path where model weights are saved                  |

This configuration file allows for easy adjustments to the model's parameters, enabling fine-tuning and optimization according to specific requirements or dataset characteristics.

When documenting the training process, it’s beneficial to keep instructions straightforward and clear, especially for users who may not be familiar with running Python scripts. Here’s how you can present the training instructions to align with common GitHub README practices:


### Training

To initiate the training process for the model, you only need to execute the `train.py` script from your command line or terminal. This simplicity is possible because all necessary model parameters have been pre-configured in the `utils/config.py` file, ensuring a smooth start without the need for initial adjustments.

Here’s how you can start the training:

```bash
python train.py
```

Upon running this command, the model will begin training using the configurations specified in `utils/config.py`. Make sure you have followed the installation and configuration steps before proceeding to ensure that all dependencies are properly installed and configured.


### Prediction

To predict or summarize text using our command-line tool, follow the syntax provided below. Replace `<text_to_summarize>` with the text you wish to summarize.

```bash
python summarize.py --input "<text_to_summarize>"
```

For instance, if you want to summarize "Hello, world! This is a test sentence.", the command would look like this:

```bash
python summarize.py --input "Hello, world! This is a test sentence."
```

Make sure to include the text within quotation marks if it contains spaces or special characters.
