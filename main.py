import torch
import config
import src.CNN
import src.utils
import src.autoencoder
import src.dataset
from matplotlib import pyplot as plt
from src.visualisation import plot_counterfactual_grid
from torchvision import datasets, transforms


def main():
    device = src.utils.get_device()
    print(f"Using device: {device}")

    # --- Load the trained black-box classifier ---
    model = src.CNN.MnistCNN().to(device)
    src.utils.load_model(model)
    model.eval()

    # --- Load the trained class-conditional plausibility autoencoders ---
    plausibility_model = src.autoencoder.ClassConditionalPlausibility()
    plausibility_model.load_saved_autoencoders()

    # --- Load test dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.MNIST_MEAN,), (config.MNIST_STD,))
    ])
    dataset = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=transform)

    # --- Pick an example and a target class ---
    number = 231
    image, label = dataset[number]
    image = image.unsqueeze(0).to(device)  # (1,1,28,28)
    target_class = 8

    print(f"Original label: {label}, target class: {target_class}")

    # --- Generate and visualize the counterfactual ---
    examples = [(image, target_class)]
    all_metrics, summary = plot_counterfactual_grid(
        model, plausibility_model, examples, device=device,
        generate_kwargs={"num_steps": 300, "lr": 0.01},
        save_path=str(config.IMAGE_SAVE_DIR / f"counterfactual_{number}.png")
    )

    print(summary)

## IGNORE THE COMMENT BELOW
"""def old_nasty_main_that_shall_not_be_used_again_but_lets_leave_it_just_in_case():
    #src.trainer.run_training();

    model = src.CNN.MnistCNN();
    src.utils.load_model( model ) # i get some saved model
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(config.DATA_DIR, train=False,  download=True, transform=transform) # I load the dataset
    number = 3000 # random number
    image, label = dataset[number]
    #src.trainer.save_model_visual_result( model , image.unsqueeze(0) , "img " + str(number) ) #unsqueeze bc (1,1,28,28) is required
    #plausability = src.autoencoder.ClassConditionalPlausibility()
    #train , _ = get_dataloaders()
    #plausability.load_saved_autoencoders()
    #plausability.train_all( train , src.utils.get_device() );
    #plausability.save_trained_autoencoders()
    image = image.unsqueeze(0) # one_image_saliency requires tensor(1,1,28,28) and image is (1,28,28)
    for i in range(5):
        saliency_map = src.saliency.one_image_saliency( model , image , 8 ) # gerenate saliency map
        src.saliency.save_saliency_result( image , saliency_map , "saliency_" + str(number) + "_" + str(i+1) ) # save the map
        image = src.utils.merge_image_saliency( image , saliency_map ) # make new image by adding saliency
        if( image.dim() == 3 ):
            image = image.unsqueeze(0)
    print( model(image) )"""
## IGNORE THE COMMENT ABOVE

def retrain_plausibility():
    plausibility = src.autoencoder.ClassConditionalPlausibility()
    train, _ = src.dataset.get_dataloaders()
    plausibility.load_saved_autoencoders() # IM NOT SURE IF LOADING WORKS
    plausibility.train_all( train , src.utils.get_device() )
    plausibility.save_trained_autoencoders()
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(config.DATA_DIR, train=False,  download=True, transform=transform) # I load the dataset
    number = 3000 # random number
    image, label = dataset[number]
    with torch.no_grad():
        new_image = plausibility.autoencoders[ 6 ]( image )
        broken_image = plausibility.autoencoders[ 7 ]( image )
        fig, axes = plt.subplots(1, 3, figsize=(6, 3))
        axes[0].imshow( image.squeeze().detach().numpy() )
        axes[1].imshow( new_image.squeeze().detach().numpy() )
        axes[2].imshow( broken_image.squeeze().detach().numpy() )
        src.utils.save_image( plt , "test_of_autoencoder" )


if __name__ == "__main__":
    main()