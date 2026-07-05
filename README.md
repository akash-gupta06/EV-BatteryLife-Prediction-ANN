# NASA Battery Life Prediction

This project uses a Neural Network (Keras) to predict **ambient temperature** during battery testing, based on battery health parameters like Capacity, Resistance (Re), and Charge Transfer Resistance (Rct).

## What this project does

1. **Loads Data**: Reads NASA battery test data (`NASA-battery-data.csv`).
2. **Cleans Data**:
   - Drops irrelevant columns (IDs, filenames, timestamps).
   - Converts text columns to numeric values.
   - Fills missing values using column mean.
3. **Encodes Data**: Converts `type` (charge/discharge/impedance) into numbers using Label Encoding.
4. **Splits Data**: 80% training, 20% testing.
5. **Scales Data**: Uses MinMaxScaler to normalize input features.
6. **Builds Model**: A neural network with:
   - Input layer + Hidden layer (64 units)
   - Dropout layer (prevents overfitting)
   - Hidden layer (32 units)
   - Dropout layer
   - Output layer (1 unit — predicts temperature)
7. **Trains Model**: 200 epochs, batch size 32.
8. **Evaluates Model**: Plots training vs validation loss.
9. **Predicts**: Takes battery type, Capacity, Re, and Rct as input, predicts ambient temperature.
10. **Saves Model**: Model, scaler, and label encoder saved as `.pkl` and `.h5` files for future use.

## Files Used

| File | Purpose |
|------|---------|
| `NASA-battery-data.csv` | Raw battery test data (input) |
| `battery_life_model.h5` | Saved trained model |
| `scaler.pkl` | Saved MinMaxScaler |
| `label_encoder.pkl` | Saved LabelEncoder |

## Tools Used

- **Python**
- **Pandas** — data handling
- **Scikit-learn** — preprocessing, train/test split
- **TensorFlow / Keras** — building and training neural network
- **Matplotlib** — visualizing loss

## How to Run

1. Place `NASA-battery-data.csv` in the project folder.
2. Run the notebook step by step: load → clean → encode → train → evaluate → predict.
3. Use `predict_battery_life()` function to test predictions with your own values.

## Example Prediction

```python
predict_battery_life(
    type_discharge='charge',
    Capacity=1.674305,
    Re=-2.9765e11,
    Rct=1.055903e12,
    label_encoder=label_encoder,
    scaler=scaler,
    model=model
)
```

## Notes

- Missing values in `Capacity`, `Re`, and `Rct` are filled with the column mean.
- Model is a simple regression-style neural network (`linear` output activation) since it predicts a continuous temperature value.
- Validation loss is higher than training loss — model may be slightly overfitting; consider more data or regularization for improvement.
