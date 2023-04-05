import torch
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc


class TrainerTester:

    @staticmethod
    def train_loop(model, optimizer, X, y, lr, ud, batch_size=32, iterations=1000):
        lossi = []
        for k in range(iterations):
            batch_indexes = torch.randint(0, X.shape[0], (batch_size,))
            X_batch, y_batch = X[batch_indexes], y[batch_indexes] # batch X,Y
            pred, loss, out_values = model(X_batch, y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #tracking
            lossi.append(loss.item())

            with torch.no_grad():
                if k % 100 == 99:
                    print(f"loss: {loss} iteration: {k}/{iterations}")
                    plt.plot(torch.tensor(lossi).view(-1, 10).mean(dim=1).log10().tolist())
                    plt.show()

                ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in model.parameters()])

        return lossi

    @staticmethod
    def test_loop(model, Xtest, ytest):
        model.eval()
        X_batch, y_batch = Xtest, ytest

        with torch.no_grad():
            pred, loss, out_values = model(X_batch, y_batch)
            correct = (pred.argmax(1) == y_batch).type(torch.float32).sum().item() / y_batch.shape[0]
            kappa = cohen_kappa_score(pred.argmax(1), y_batch)

        print(f"Test loss: {loss:>8f} \n Accuracy: {correct:>8f} \n kappa: {kappa} \n")
        model.train()
        return out_values

    @staticmethod
    def test_and_show(model, Xtest, ytest):
        model.eval()
        X_batch, y_batch = Xtest, ytest
        with torch.no_grad():
            pred, loss, out_values = model(X_batch, y_batch)
            accuracy = pred.argmax(1) == y_batch
            model.train()
            return accuracy