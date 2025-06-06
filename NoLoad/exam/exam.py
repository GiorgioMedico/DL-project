import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# -------------------- Constants (Provided Framework) --------------------
OPERATORS = ['+', '-', '*', '/']
IDENTIFIERS = list('abcde')
SPECIAL_TOKENS = ['PAD', 'SOS', 'EOS']
SYMBOLS = ['(', ')', '+', '-', '*', '/']
VOCAB = SPECIAL_TOKENS + SYMBOLS + IDENTIFIERS + ['JUNK']

token_to_id = {tok: i for i, tok in enumerate(VOCAB)}
id_to_token = {i: tok for tok, i in token_to_id.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = token_to_id['PAD']
EOS_ID = token_to_id['EOS']
SOS_ID = token_to_id['SOS']

MAX_DEPTH = 3
MAX_LEN = 4*2**MAX_DEPTH - 2

print("="*60)
print("INFIX TO POSTFIX NEURAL NETWORK TRANSLATION")
print("="*60)
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Maximum sequence length: {MAX_LEN}")
print(f"Vocabulary: {VOCAB}")

# -------------------- Expression Generation (Provided Core Logic) --------------------
def generate_infix_expression(max_depth):
    if max_depth == 0:
        return random.choice(IDENTIFIERS)
    elif random.random() < 0.5:
        return generate_infix_expression(max_depth - 1)
    else:
        left = generate_infix_expression(max_depth - 1)
        right = generate_infix_expression(max_depth - 1)
        op = random.choice(OPERATORS)
        return f'({left} {op} {right})'

def tokenize(expr):
    return [c for c in expr if c in token_to_id]

def infix_to_postfix(tokens):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output, stack = [], []
    for token in tokens:
        if token in IDENTIFIERS:
            output.append(token)
        elif token in OPERATORS:
            while stack and stack[-1] in OPERATORS and precedence[stack[-1]] >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
    while stack:
        output.append(stack.pop())
    return output

def encode(tokens, max_len=MAX_LEN):
    ids = [token_to_id[t] for t in tokens] + [EOS_ID]
    return ids + [PAD_ID] * (max_len - len(ids))

def decode_sequence(token_ids, id_to_token, pad_token='PAD', eos_token='EOS'):
    """
    Converts a list of token IDs into a readable string by decoding tokens.
    Stops at the first EOS token if present, and ignores PAD tokens.
    """
    tokens = []
    for token_id in token_ids:
        token = id_to_token.get(token_id, '?')
        if token == eos_token:
            break
        if token != pad_token:
            tokens.append(token)
    return ' '.join(tokens)

def generate_dataset(n, max_depth=MAX_DEPTH):
    X, Y = [], []
    for _ in range(n):
        expr = generate_infix_expression(max_depth)
        infix = tokenize(expr)
        postfix = infix_to_postfix(infix)
        X.append(encode(infix))
        Y.append(encode(postfix))
    return np.array(X), np.array(Y)

def shift_right(seqs):
    shifted = np.zeros_like(seqs)
    shifted[:, 1:] = seqs[:, :-1]
    shifted[:, 0] = SOS_ID
    return shifted

# -------------------- Model Architecture --------------------
def create_efficient_encoder_decoder_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embedding_dim=128, hidden_dim=256):
    """
    Encoder-Decoder LSTM model optimized for parameter efficiency
    Target: Under 2M parameters while maintaining strong performance
    """

    # Encoder
    encoder_input = layers.Input(shape=(max_len,), name='encoder_input')
    encoder_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_input)

    # Single bidirectional LSTM for encoder efficiency
    encoder_lstm = layers.Bidirectional(
        layers.LSTM(hidden_dim//2, return_state=True, dropout=0.2, recurrent_dropout=0.2)
    )(encoder_embedding)

    encoder_outputs = encoder_lstm[0]
    encoder_state_h = layers.Concatenate()([encoder_lstm[1], encoder_lstm[3]])
    encoder_state_c = layers.Concatenate()([encoder_lstm[2], encoder_lstm[4]])
    encoder_states = [encoder_state_h, encoder_state_c]

    # Decoder
    decoder_input = layers.Input(shape=(max_len,), name='decoder_input')
    decoder_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_input)

    # Single LSTM decoder with enhanced capacity
    decoder_lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True,
                              dropout=0.2, recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Output projection with intermediate layer
    dense_intermediate = layers.Dense(hidden_dim//2, activation='relu')(decoder_outputs)
    dense_intermediate = layers.Dropout(0.3)(dense_intermediate)
    output = layers.Dense(vocab_size, activation='softmax')(dense_intermediate)

    model = models.Model(inputs=[encoder_input, decoder_input], outputs=output)
    return model

# -------------------- Training Strategy --------------------
def create_and_compile_model():
    """Create model ensuring parameter count is under 2M"""
    model = create_efficient_encoder_decoder_model()

    param_count = model.count_params()
    print(f"\nModel parameter count: {param_count:,}")

    # Compile with appropriate loss and metrics
    model.compile(
        optimizer=Adam(learning_rate=0.002, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(train_size=10000, val_size=1000, batch_size=64, epochs=50, 
                learning_rate=0.002, clipnorm=1.0, early_stopping_patience=8, 
                lr_reduction_factor=0.7, lr_reduction_patience=4, min_lr=1e-6,
                verbose=1):
    
    # Create separators
    main_separator = "=" * 60
    sub_separator = "-" * 40
    
    print(f"\n{main_separator}")
    print("TRAINING PHASE")
    print(main_separator)
    
    print("\nGenerating datasets...")
    print(f"  Training samples: {train_size:,}")
    print(f"  Validation samples: {val_size:,}")
    
    X_train, Y_train = generate_dataset(train_size)
    decoder_input_train = shift_right(Y_train)
    
    X_val, Y_val = generate_dataset(val_size)
    decoder_input_val = shift_right(Y_val)
    
    # Model creation section
    print(f"\n{sub_separator}")
    print("MODEL ARCHITECTURE")
    print(sub_separator)
    model = create_and_compile_model()
    model.summary()
    
    print(f"\n{sub_separator}")
    print("TRAINING CONFIGURATION")
    print(sub_separator)
    
    # Create callbacks with configurable parameters
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=early_stopping_patience, 
            restore_best_weights=True, 
            verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=lr_reduction_factor, 
            patience=lr_reduction_patience, 
            min_lr=min_lr, 
            verbose=verbose
        )
    ]
    
    # Display all configuration parameters
    print(f"Optimizer: Adam (lr={learning_rate}, clipnorm={clipnorm})")
    print("Loss function: sparse_categorical_crossentropy")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print("Callbacks:")
    print(f"  - Early stopping: patience={early_stopping_patience}, monitor=val_loss")
    print(f"  - Learning rate reduction: factor={lr_reduction_factor}, patience={lr_reduction_patience}")
    print(f"  - Minimum learning rate: {min_lr}")
    
    print(f"\n{sub_separator}")
    print("TRAINING EXECUTION")
    print(sub_separator)
    
    # Training with all configurable parameters
    history = model.fit(
        [X_train, decoder_input_train], Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_val, decoder_input_val], Y_val),
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Training analysis section
    print(f"\n{sub_separator}")
    print("TRAINING RESULTS")
    print(sub_separator)
    
    # Extract training metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    epochs_trained = len(history.history['loss'])
    
    # Display results with improved formatting
    print(f"Training completed after {epochs_trained} epochs")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    return model, history

# -------------------- Autoregressive Generation --------------------
def autoregressive_decode(model, encoder_input, max_length=MAX_LEN):
    """
    Autoregressive generation as required for encoder-decoder models
    """
    encoder_input = np.expand_dims(encoder_input, 0)

    # Initialize decoder input with SOS token
    decoder_input = np.zeros((1, max_length), dtype=np.int32)
    decoder_input[0, 0] = SOS_ID

    generated_sequence = [SOS_ID]

    for i in range(1, max_length):
        # Get predictions from model
        predictions = model.predict([encoder_input, decoder_input], verbose=0)

        # Select next token (greedy decoding)
        next_token = np.argmax(predictions[0, i-1, :])

        # Update decoder input and generated sequence
        decoder_input[0, i] = next_token
        generated_sequence.append(next_token)

        # Stop if EOS token is generated
        if next_token == EOS_ID:
            break

    # Pad to max_length
    while len(generated_sequence) < max_length:
        generated_sequence.append(PAD_ID)

    return np.array(generated_sequence)

# -------------------- Evaluation (Exact Specification) --------------------
def prefix_accuracy_single(y_true, y_pred, id_to_token, eos_id=EOS_ID, verbose=False):
    t_str = decode_sequence(y_true, id_to_token).split(' EOS')[0]
    p_str = decode_sequence(y_pred, id_to_token).split(' EOS')[0]
    t_tokens = t_str.strip().split()
    p_tokens = p_str.strip().split()
    max_len = max(len(t_tokens), len(p_tokens))

    match_len = sum(x == y for x, y in zip(t_tokens, p_tokens))
    score = match_len / max_len if max_len>0 else 0

    if verbose:
        print("TARGET :", ' '.join(t_tokens))
        print("PREDICT:", ' '.join(p_tokens))
        print(f"PREFIX MATCH: {match_len}/{len(t_tokens)} → {score:.2f}")

    return score

def test(model, no=20, rounds=10):
    print(f"Evaluating model performance on {no} expressions × {rounds} rounds...")
    rscores = []
    for i in range(rounds):
        print(f"Round {i+1}/{rounds}...")
        X_test, Y_test = generate_dataset(no)
        scores = []
        for j in range(no):
            encoder_input = X_test[j]
            generated = autoregressive_decode(model, encoder_input)[1:]  # remove SOS
            scores.append(prefix_accuracy_single(Y_test[j], generated, id_to_token))
        round_mean = np.mean(scores)
        rscores.append(round_mean)
        print(f"  Round {i+1} accuracy: {round_mean:.4f}")
    
    final_mean = np.mean(rscores)
    final_std = np.std(rscores)
    print("\nEvaluation complete!")
    print(f"Mean accuracy across all rounds: {final_mean:.4f}")
    print(f"Standard deviation: {final_std:.4f}")
    
    return final_mean, final_std

# -------------------- Visualization and Analysis --------------------
def plot_training_history(history):
    
    # Extract history data
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(loss_history, label='Training Loss', linewidth=2, color='blue')
    ax1.plot(val_loss_history, label='Validation Loss', linewidth=2, color='red')
    ax1.set_title('Loss During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot  
    ax2.plot(acc_history, label='Training Accuracy', linewidth=2, color='green')
    ax2.plot(val_acc_history, label='Validation Accuracy', linewidth=2, color='orange')
    ax2.set_title('Accuracy During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print training summary
    epochs_trained = len(loss_history)
    final_train_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]
    final_train_acc = acc_history[-1]
    final_val_acc = val_acc_history[-1]
    best_val_acc = max(val_acc_history)
    
    print(f"\n{'-'*50}")
    print("TRAINING SUMMARY")
    print(f"{'-'*50}")
    print(f"Total epochs trained: {epochs_trained}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

def demonstrate_model_performance(model, num_examples=10):
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE DEMONSTRATION")
    print(f"{'='*70}")

    X_demo, Y_demo = generate_dataset(num_examples)

    perfect_matches = 0
    partial_matches = 0
    total_score = 0

    print(f"Testing on {num_examples} randomly generated expressions:\n")

    for i in range(num_examples):
        encoder_input = X_demo[i]
        target = Y_demo[i]
        generated = autoregressive_decode(model, encoder_input)[1:]

        infix_str = decode_sequence(encoder_input, id_to_token)
        target_str = decode_sequence(target, id_to_token)
        generated_str = decode_sequence(generated, id_to_token)

        score = prefix_accuracy_single(target, generated, id_to_token)
        total_score += score

        if score == 1.0:
            perfect_matches += 1
            status = "✓ PERFECT"
        elif score > 0.5:
            partial_matches += 1
            status = "~ PARTIAL"
        else:
            status = "✗ POOR"

        print(f"Example {i+1:2d}:")
        print(f"  Input (Infix):     {infix_str}")
        print(f"  Target (Postfix):  {target_str}")
        print(f"  Generated:         {generated_str}")
        print(f"  Score: {score:.3f}  [{status}]")
        print()

    avg_score = total_score / num_examples
    
    print(f"{'-'*50}")
    print("DEMONSTRATION RESULTS")
    print(f"{'-'*50}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Perfect matches: {perfect_matches}/{num_examples} ({perfect_matches/num_examples*100:.1f}%)")
    print(f"Partial matches: {partial_matches}/{num_examples} ({partial_matches/num_examples*100:.1f}%)")
    print(f"Poor matches: {num_examples - perfect_matches - partial_matches}/{num_examples} ({(num_examples - perfect_matches - partial_matches)/num_examples*100:.1f}%)")

# -------------------- Main Execution --------------------
def main():
    """Main execution"""

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Training phase with documentation
    model, history = train_model()

    # Visualize training history
    print(f"\n{'='*60}")
    print("TRAINING HISTORY VISUALIZATION")
    print(f"{'='*60}")
    plot_training_history(history)

    # Demonstrate model performance
    demonstrate_model_performance(model)

    # Final evaluation as specified
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    result_mean, result_std = test(model, no=20, rounds=10)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Mean Score: {result_mean:.4f}")
    print(f"Standard Deviation: {result_std:.4f}")
    print(f"Model Parameters: {model.count_params():,}")

    return model, history, result_mean, result_std

# Execute main function
if __name__ == "__main__":
    main()