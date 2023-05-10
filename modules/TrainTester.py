import torch
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc


import torch
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc


class TrainerTester:

    @staticmethod
    def train_loop(model, optimizer, X_train, y_train, X_test, y_test, lr, ud, batch_size=32, iterations=1000):
        lossi = []
        accuracyi = []
        lossv = []
        accuracyv = []
        kappai = []
        kappav = []
        for k in range(iterations):
            batch_indexes = torch.randint(0, X_train.shape[0], (batch_size,))
            batch_test_indexes = torch.randint(0, X_test.shape[0], (batch_size,))

            X_batch, y_batch = X_train[batch_indexes], y_train[batch_indexes]  # train batch X,Y

            #data augmentation
            # X_batch, _ = transforms['freq'].operation(X_batch, None, 10., sfreq)
            # X_batch, _ = transforms['gauss'].operation(X_batch, None, std=1e-5)

            X_test_batch, y_test_batch = X_test[batch_test_indexes], y_test[batch_test_indexes]  # test batch X,Y

            pred, loss, out_values = model(X_batch,  y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                #tracking training metrics
                lossi.append(loss.item())
                accuracyi.append((pred.argmax(1) == y_batch).type(torch.float32).sum().item() / y_batch.shape[0])
                kappai.append(cohen_kappa_score(pred.argmax(1), y_batch))
                #tracking test metrics
                test_loss, test_accuracy, test_kappa = TrainerTester.test_loss(model, X_test_batch, y_test_batch)
                lossv.append(test_loss)
                accuracyv.append(test_accuracy)
                kappav.append(test_kappa)

                if (k+1) % 100 == 0:
                    print(f"loss: {loss} iteration: {k+1}/{iterations}")
                    plt.title('loss')
                    plt.plot(torch.tensor(lossv).view(-1, 10).mean(dim=1).tolist(), label="test loss")
                    plt.plot(torch.tensor(lossi).view(-1, 10).mean(dim=1).tolist(), label="train loss")
                    plt.show()

                    plt.title('accuracy')
                    plt.plot(torch.tensor(accuracyv).view(-1, 10).mean(dim=1).tolist(), label="test accuracy")
                    plt.plot(torch.tensor(accuracyi).view(-1, 10).mean(dim=1).tolist(), label="train accuracy")
                    plt.show()

                    plt.title('kappa')
                    plt.plot(torch.tensor(kappav).view(-1, 10).mean(dim=1).tolist(), label="test accuracy")
                    plt.plot(torch.tensor(kappai).view(-1, 10).mean(dim=1).tolist(), label="train accuracy")
                    plt.show()

                ud.append([((lr * p.grad).std() / p.data.std()).item() for p in model.parameters()])

        return lossi

    @staticmethod
    def test_loss(model, X_test,  y_test):
        model.eval()
        with torch.no_grad():
            pred, loss, out_values = model(X_test, y_test)
            accuracy = (pred.argmax(1) == y_test).type(torch.float32).sum().item() / y_test.shape[0]
            kappa = cohen_kappa_score(pred.argmax(1), y_test)
        model.train()
        return loss, accuracy, kappa

    @staticmethod
    def test_loop(model, X_test,  y_test):
        model.eval()
        with torch.no_grad():
            pred, loss, out_values = model(X_test, y_test)
            accuracy = (pred.argmax(1) == y_test).type(torch.float32).sum().item() / y_test.shape[0]
            kappa = cohen_kappa_score(pred.argmax(1), y_test)

        print(f"Test loss: {loss:>8f} \n Accuracy: {accuracy:>8f} \n kappa: {kappa} \n")
        model.train()
        return out_values

    @staticmethod
    def test_and_show(model, X_test,  y_test):
        model.eval()
        with torch.no_grad():
            pred, loss, out_values = model(X_test,  y_test)
            accuracy = pred.argmax(1) == y_test
            model.train()
            return accuracy

