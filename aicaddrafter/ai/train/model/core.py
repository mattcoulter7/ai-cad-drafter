from keras.layers import (
    Dense,
    Input,
    LSTM,
    Dense,
    Embedding,
    TimeDistributed,
    Concatenate,
    Attention
)
from keras.models import (
    Model,
    load_model
)


def get_seq2seq(n_input_features, n_output_features, n_units):
    # Input tensor for the encoder
    encoder_inputs = Input(shape=(None, n_input_features))

    # LSTM encoder
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # Discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder. The decoder will use `encoder_states` as its initial state.
    decoder_inputs = Input(shape=(None, n_output_features))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output_features, activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Create the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def load_models(fp, config):
    model = load_model(fp)
    # Detailed layer extraction depends on saved model structure
    encoder_inputs = model.input[0]
    _, state_h_enc, state_c_enc = model.layers[2].output
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    decoder_inputs = Input(shape=(None, config["n_output_features"]))
    decoder_state_input_h = Input(shape=(config["n_units"],))
    decoder_state_input_c = Input(shape=(config["n_units"],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]   
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return encoder_model, decoder_model
