"""
Script to generate sample sentiment analysis dataset.
Creates a CSV file with text and sentiment labels for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Sample texts with known sentiments
positive_texts = [
    "This product is amazing! I love it so much.",
    "Excellent quality and fast shipping. Highly recommend!",
    "Best purchase I've made this year. Worth every penny.",
    "Outstanding service and great customer support.",
    "Perfect! Exactly what I was looking for.",
    "Fantastic product, exceeded my expectations.",
    "Great value for money. Very satisfied!",
    "Wonderful experience, will definitely buy again.",
    "Top quality product, very happy with my purchase.",
    "Amazing features and great design. Love it!",
    "This is the best product I've ever used.",
    "Excellent quality, fast delivery, great price.",
    "Highly satisfied with this purchase.",
    "Outstanding product, highly recommended!",
    "Perfect product for my needs. Very happy!",
    "Great product, excellent customer service.",
    "Love this product! It's exactly what I needed.",
    "Fantastic quality, very pleased with purchase.",
    "Excellent value, would buy again.",
    "Amazing product, exceeded all expectations.",
    "This is wonderful! I'm very satisfied.",
    "Great quality and fast shipping.",
    "Perfect product, highly recommend!",
    "Excellent service and great product.",
    "Outstanding quality, very happy customer.",
]

negative_texts = [
    "Terrible product, complete waste of money.",
    "Poor quality and slow shipping. Very disappointed.",
    "Worst purchase ever. Do not recommend.",
    "Bad service and unhelpful customer support.",
    "Awful product, nothing works as advertised.",
    "Horrible experience, will never buy again.",
    "Cheap quality, broke after one use.",
    "Very disappointed with this purchase.",
    "Poor quality product, not worth the price.",
    "Terrible customer service, avoid this company.",
    "This product is a complete disaster.",
    "Worst quality I've ever seen.",
    "Very unhappy with this purchase.",
    "Poor product, does not meet expectations.",
    "Terrible experience, would not recommend.",
    "Bad quality, waste of money.",
    "Horrible product, completely unsatisfied.",
    "Poor service and terrible product.",
    "Awful quality, very disappointed.",
    "This is terrible, do not buy.",
    "Worst product I've ever purchased.",
    "Poor quality, not recommended.",
    "Terrible experience, very unhappy.",
    "Bad product, complete waste.",
    "Horrible quality, avoid at all costs.",
]

neutral_texts = [
    "The product arrived on time. It works as expected.",
    "Average product, nothing special but it works.",
    "It's okay, does what it's supposed to do.",
    "Standard quality product, meets basic requirements.",
    "The product is fine, nothing exceptional.",
    "It works, but could be better.",
    "Average quality, acceptable for the price.",
    "The product is functional, nothing more.",
    "It's decent, nothing to complain about.",
    "Standard product, does its job.",
    "The item arrived as described.",
    "It's okay, nothing special.",
    "Average product, meets expectations.",
    "Functional product, nothing exceptional.",
    "It works fine, standard quality.",
    "The product is acceptable.",
    "Average quality, nothing remarkable.",
    "It does what it's supposed to do.",
    "Standard product, works as expected.",
    "The item is fine, nothing special.",
    "It's okay, meets basic needs.",
    "Average product, acceptable quality.",
    "Functional, nothing to write home about.",
    "Standard quality, works fine.",
    "The product is decent, nothing exceptional.",
]

def generate_sample_dataset(output_path: str = "data/raw/sample_sentiment_data.csv", num_samples: int = 1000):
    """
    Generate a sample sentiment analysis dataset.
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Number of samples to generate
    """
    # Create all texts list
    all_texts = []
    all_labels = []
    
    # Add positive texts
    for text in positive_texts:
        all_texts.append(text)
        all_labels.append("positive")
    
    # Add negative texts
    for text in negative_texts:
        all_texts.append(text)
        all_labels.append("negative")
    
    # Add neutral texts
    for text in neutral_texts:
        all_texts.append(text)
        all_labels.append("neutral")
    
    # Generate more samples by adding variations
    np.random.seed(42)
    
    # Create variations
    variations = [
        "Really ", "Very ", "Extremely ", "Quite ", "Pretty ",
        "", "", "", ""  # Some without modification
    ]
    
    while len(all_texts) < num_samples:
        # Randomly select a base text
        idx = np.random.randint(0, len(positive_texts) + len(negative_texts) + len(neutral_texts))
        
        if idx < len(positive_texts):
            base_text = positive_texts[idx]
            label = "positive"
        elif idx < len(positive_texts) + len(negative_texts):
            base_text = negative_texts[idx - len(positive_texts)]
            label = "negative"
        else:
            base_text = neutral_texts[idx - len(positive_texts) - len(negative_texts)]
            label = "neutral"
        
        # Add variation
        variation = np.random.choice(variations)
        text = variation + base_text.lower()
        
        all_texts.append(text)
        all_labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts[:num_samples],
        'label': all_labels[:num_samples]
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print(f"\nDataset saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate dataset
    df = generate_sample_dataset(num_samples=1000)
    print("\nSample data:")
    print(df.head(10))

