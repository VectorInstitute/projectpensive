from model import CivilityModel

if __name__ == "__main__":

    civility_model = CivilityModel()
    history = civility_model.train(
        epochs=5,
        batch_size=100
    )
    civility_model.model.save("civility_model2")
