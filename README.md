# Simulation of Federated Learning in Hierarchial Networks

- **Project Type**: Open-Ended Lab Project
- **Contributors**: [Bharath Kallakuri](https://github.com/BharathKallakuri), [K Srirama Srikar](https://github.com/k-srirama-srikar)
- **Mentors**: Dr. Anoop Thomas, Dr. Birenjith Sashidharan
- **Semester**: Fall 2025

> We were awarded an S grade (10/10) for the work done in this project!

## About the Github Repository

* The repository consists of codes for Simulation of Federated Learning in Hierarchical setup.
* We have implemented Repetition coding and Aligned MDS (Maximum Distance Sperable) Coding technique for communication.
* We have used FEMNIST Dataset and Logistic Regression and CNN for training and testing.
* Heterogeneous Setup for AMC, which can have different clients with different erasure tolerance.
<!-- ### Code Overview:
* There are three classes Client, Helper and Master.
* Client has the function `train_local` to train on the data it has locally and sends the model to the helpers.
* Master has `run_round` function that could be used to run one round in all the clients, then the clients run the training locally based on the dataset they have, and send the weights to the helper.
* Helper receives the client model weights from the helpers.
* Helpers class is inherited from Dictionary class, with an additional attribute `hid`, which is the helper id of the helper.
* In AMC scheme, the client splits the payload into pieces and then applies the MDS Code on it and then sends them to the helpers, where as in ARC Scheme the clients send the whole payload to the multiple helpers based on the erasure count.
* In the ARC code, the above thing is achieved by sending the model to every helper and randomly making the values in helpers for (nh-s-1) helpers to None, thus simulating the erasures in communication.
* The erasures are simulated based on the Failure Matrix generated.
* The dataset has loader class that can be used to do grouping based on the `writer_id` and group size, to allocate data of a group of writers to the clients.
* There is another run_round function commented in the Master class which can be used for validation of the AMC/ARC setup with the setup without the helper nodes. -->

## Code Overview:
1. **`Client`**

   * Represents a federated client with access to a local subset of the dataset.
   * **`train_local`**: Trains the client’s model locally using its dataset.
   * Sends model updates to helper nodes.
   * In the **AMC scheme**, the client splits the model payload into multiple pieces, applies the **MDS coding**, and sends the encoded pieces to the helpers.
   * In the **ARC scheme**, the client sends the **whole payload** to multiple helpers, with erasures simulated by randomly setting some helper entries to `None`.

2. **`Helper`**

   * Receives model updates from clients.
   * Inherits from Python’s **`OrderedDict`** and has an additional attribute **`hid`** representing the helper ID.
   * Stores encoded or full client updates, depending on the scheme (AMC or ARC).

3. **`Master`**

   * Orchestrates the federated learning process.
   * **`run_round`**: Executes a full round of training:

     1. Sends the current global model to all clients.
     2. Clients train locally and send model updates to helpers.
     3. Aggregates updates from helpers to reconstruct the new global model.
   * **Commented `run_round` for validation**:There is another `run_round` function that is commented which can be used for validation of the setup with the setup without the helper nodes.



### **Communication and Erasure Handling**

* **Failure Matrix**: Simulates communication failures between clients and helpers.
* **AMC Scheme**:

  * Clients split their model payload into pieces.
  * Apply **MDS coding** to create redundant pieces.
  * Send coded pieces to helpers.
* **ARC Scheme**:

  * Clients send the full payload to multiple helpers.
  * Randomly set some helper entries to `None` to simulate erasures.


### **Dataset Handling**

* **Loader Class (`OptimizedFemnistLoader`)**:
  * Groups data by `writer_id` and allocates a **group of writers** to each client.
  * Efficiently indexes dataset to allow fast selection of client-specific data.
* Supports batched training via PyTorch `DataLoader`.


1. Load and partition dataset among clients.
2. Initialize **Client**, **Helper**, and **Master** objects.
3. For each FL round:

   * **Master** sends the global model to all clients.
   * **Clients** train locally and send updates to helpers (AMC or ARC scheme).
   * **Helpers** store received updates.
   * **Master** reconstructs the global model from helper updates.
   * Evaluate the global model on the test dataset.
4. Track communication payloads and accuracy per round.


