import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess
import matplotlib.pyplot as plt

def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        input_encoding = question + " [SEP] " + passage

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }


def evaluate_model(model, dataloader, device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, device, lr):

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()

    train_acc_list = [] # to store training accuracies for each epoch
    val_acc_list = []

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device) 
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = mymodel(input_ids, attention_mask=attention_mask).logits

            model_loss = loss(logits, labels)

             # compute the gradients
            model_loss.backward()

            # update the model parameters
            optimizer.step()

            # update the learning rate
            lr_scheduler.step()

            # reset the gradients for the next iteration
            optimizer.zero_grad()
            

            predictions = torch.argmax(logits, dim=1)

            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=batch['labels'])

        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")

        # train_acc = train_accuracy.compute()
        train_acc = train_accuracy.compute()['accuracy']

        print(f" - Average training metrics: accuracy={train_acc}")

        train_acc_list.append(train_acc)

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        val_acc_list.append(val_accuracy['accuracy'])
    
    # plot and save the training accuracy as a function of training epochs
    # plt.plot(range(1, num_epochs+1), train_acc_list)
    # plt.plot(range(1, num_epochs+1), val_acc_list)
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig('accuracy.png')
    # plt.show()

# implements parameter selection and returns a list with (learning rate, epochs, validation accuracy, test accuracy)
def train2(mymodel, train_dataloader, validation_dataloader, device):

    learning_rates = [1e-4, 1e-3]
    num_epochs_list = [7, 9]

    results = []

    for lr in learning_rates:
        for num_epochs in num_epochs_list:

            # here, we use the AdamW optimizer. Use torch.optim.Adam.
            # instantiate it on the untrained model parameters with a learning rate of lr
            print(f" >>>>>>>>  Initializing optimizer with lr={lr}")
            optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

            # now, we set up the learning rate scheduler
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=50,
                num_training_steps=len(train_dataloader) * num_epochs
            )

            loss = torch.nn.CrossEntropyLoss()

            train_acc_list = [] # to store training accuracies for each epoch
            val_acc_list = []

            for epoch in range(num_epochs):

                # put the model in training mode (important that this is done each epoch,
                # since we put the model into eval mode during validation)
                mymodel.train()

                # load metrics
                train_accuracy = evaluate.load('accuracy')

                print(f"Epoch {epoch + 1} training:")

                for i, batch in enumerate(train_dataloader):

                    input_ids = batch['input_ids'].to(device) 
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    logits = mymodel(input_ids, attention_mask=attention_mask).logits

                    model_loss = loss(logits, labels)

                     # compute the gradients
                    model_loss.backward()

                    # update the model parameters
                    optimizer.step()

                    # update the learning rate
                    lr_scheduler.step()

                    # reset the gradients for the next iteration
                    optimizer.zero_grad()


                    predictions = torch.argmax(logits, dim=1)

                    # update metrics
                    train_accuracy.add_batch(predictions=predictions, references=batch['labels'])

                # print evaluation metrics
                print(f" ===> Epoch {epoch + 1}")

                # train_acc = train_accuracy.compute()
                train_acc = train_accuracy.compute()['accuracy']

                print(f" - Average training metrics: accuracy={train_acc}")

                train_acc_list.append(train_acc)

                # normally, validation would be more useful when training for many epochs
                val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
                print(f" - Average validation metrics: accuracy={val_accuracy}")
                val_acc_list.append(val_accuracy['accuracy'])

                # evaluate the model on the test set and store the results
                test_accuracy = evaluate_model(mymodel, test_dataloader, device)
                print(f" - Test metrics: accuracy={test_accuracy}")

            results.append((lr, num_epochs, val_accuracy['accuracy'], test_accuracy['accuracy']))

    return results

def pre_process(model_name, batch_size, device, small_subset):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        dataset_dev_subset = dataset['train'][:10]
        dataset_test_subset = dataset['train'][:10]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:8000]
        dataset_dev_subset = dataset['validation']
        dataset_test_subset = dataset['train'][8000:]

    print("Size of the loaded dataset:")
    print(f" - train: {len(dataset_train_subset['passage'])}")
    print(f" - dev: {len(dataset_dev_subset['passage'])}")
    print(f" - test: {len(dataset_test_subset['passage'])}")

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--model", type=str, default="RoBERTa-base")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"

    # load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset)

    print(" >>>>>>>>  Starting training ... ")
    # train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr)
    
    # Problem 7 Parameter selection
    result = train2(pretrained_model, train_dataloader, validation_dataloader, args.device)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")

    print("(learning rate, epochs, validation accuracy, test accuracy)")

    # Problem 7 results
    print(result)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment", type=str, default=None)
#     parser.add_argument("--small_subset", action='store_true')
#     parser.add_argument("--num_epochs", type=int, default=1)
#     parser.add_argument("--lr", type=float, default=0.0001)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--models", nargs='+', default=['distilbert-base-uncased', 'RoBERTa-base'])

#     args = parser.parse_args()
#     print(f"Specified arguments: {args}")

#     assert type(args.small_subset) == bool, "small_subset must be a boolean"

#     # initialize lists to store validation and test accuracies for each model
#     val_accuracies = []
#     test_accuracies = []

#     # iterate over specified models and get validation and test accuracies for each
#     for model_name in args.models:
#         print(f"Training model {model_name}...")
#         pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(model_name,
#                                                                                                  args.batch_size,
#                                                                                                  args.device,
#                                                                                                  args.small_subset)

#         train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr)

#         val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)['accuracy']
#         val_accuracies.append(val_accuracy)
#         print(f" - Final DEV metrics: accuracy={val_accuracy}")

#         test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)['accuracy']
#         test_accuracies.append(test_accuracy)
#         print(f" - Final TEST metrics: accuracy={test_accuracy}")

#     # plot the results as a bar plot
#     fig, ax = plt.subplots()
#     bar_width = 0.4

#     x_pos = list(range(len(args.models)))
#     ax.bar(x_pos, test_accuracies, bar_width, label='Test Accuracy')
#     ax.bar([i + bar_width for i in x_pos], val_accuracies, bar_width, label='Validation Accuracy')

#     # add labels and title
#     ax.set_ylabel('Accuracy')
#     ax.set_title('Test and Validation Accuracies of Different Models')
#     ax.set_xticks([i + bar_width / 2 for i in x_pos])
#     ax.set_xticklabels(args.models, rotation=45, fontsize=7)
#     ax.legend()

#     # save the plot and show it
#     plt.savefig('test_validation_accuracies.png')
#     plt.show()
