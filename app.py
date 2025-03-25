import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import ttk, messagebox
import seaborn as sns

class BostonHousingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Boston Housing Price Prediction")
        self.root.geometry("1200x800")
        
        # Load data
        self.load_data()
        
        # Create GUI elements
        self.create_widgets()
        
        # Run initial analysis
        self.run_analysis()
    
    def load_data(self):
        """Load and preprocess the Boston housing dataset"""
        boston = fetch_openml(name='boston', version=1, as_frame=True)
        self.df = pd.DataFrame(boston.data, columns=boston.feature_names)
        self.df['PRICE'] = boston.target
        self.feature_names = boston.feature_names
        
        # Train-test split
        X = self.df.drop('PRICE', axis=1)
        y = self.df['PRICE']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    
    def create_widgets(self):
        """Create all GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Model selection
        ttk.Label(control_frame, text="Select Model:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="Both")
        ttk.Radiobutton(control_frame, text="Linear Regression", 
                        variable=self.model_var, value="Linear").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Ridge Regression", 
                        variable=self.model_var, value="Ridge").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Both Models", 
                        variable=self.model_var, value="Both").pack(anchor=tk.W)
        
        # Analysis button
        ttk.Button(control_frame, text="Run Analysis", 
                  command=self.run_analysis).pack(pady=10, fill=tk.X)
        
        # Results display
        ttk.Label(control_frame, text="Results:").pack(anchor=tk.W)
        self.results_text = tk.Text(control_frame, height=10, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Visualizations
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Correlation")
        self.notebook.add(self.tab2, text="Residuals")
        self.notebook.add(self.tab3, text="Feature Importance")
        self.notebook.add(self.tab4, text="Actual vs Predicted")
    
    def run_analysis(self):
        """Run the selected analysis and update the GUI"""
        try:
            model_choice = self.model_var.get()
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.clear_tabs()
            
            # Run selected models
            if model_choice in ["Linear", "Both"]:
                self.run_linear_regression()
            if model_choice in ["Ridge", "Both"]:
                self.run_ridge_regression()
            
            # Show correlation matrix (always)
            self.show_correlation()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def run_linear_regression(self):
        """Run linear regression and display results"""
        # Create pipeline
        lr_pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LinearRegression()
        )
        
        # Fit model
        lr_pipe.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = lr_pipe.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Display results
        self.results_text.insert(tk.END, "Linear Regression Results:\n")
        self.results_text.insert(tk.END, f"MSE: {mse:.2f}\n")
        self.results_text.insert(tk.END, f"R²: {r2:.2f}\n\n")
        
        # Store predictions for visualization
        self.lr_pred = y_pred
        
        # Show visualizations
        self.show_residuals(self.y_test, y_pred, "Linear Regression")
        self.show_feature_importance(lr_pipe.named_steps['linearregression'].coef_, 
                                   "Linear Regression Feature Importance")
        self.show_actual_vs_predicted(self.y_test, y_pred, "Linear Regression")
    
    def run_ridge_regression(self):
        """Run ridge regression and display results"""
        # Create pipeline
        ridge_pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            Ridge()
        )
        
        # Hyperparameter tuning
        param_grid = {'ridge__alpha': np.logspace(-4, 4, 20)}
        ridge_cv = GridSearchCV(ridge_pipe, param_grid, cv=5, 
                              scoring='neg_mean_squared_error', n_jobs=-1)
        ridge_cv.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = ridge_cv.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Display results
        self.results_text.insert(tk.END, "Ridge Regression Results:\n")
        self.results_text.insert(tk.END, f"MSE: {mse:.2f}\n")
        self.results_text.insert(tk.END, f"R²: {r2:.2f}\n")
        self.results_text.insert(tk.END, f"Best alpha: {ridge_cv.best_params_['ridge__alpha']:.2f}\n\n")
        
        # Store predictions for visualization
        self.ridge_pred = y_pred
        
        # Show visualizations
        self.show_residuals(self.y_test, y_pred, "Ridge Regression")
        self.show_feature_importance(ridge_cv.best_estimator_.named_steps['ridge'].coef_, 
                                   "Ridge Regression Feature Importance")
        self.show_actual_vs_predicted(self.y_test, y_pred, "Ridge Regression")
    
    def show_correlation(self):
        """Show correlation matrix in the first tab"""
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   annot_kws={"size": 8}, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Embed in Tkinter
        self.embed_plot(fig, self.tab1)
    
    def show_residuals(self, y_true, y_pred, title):
        """Show residual plot in the second tab"""
        residuals = y_true - y_pred
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title(f'{title} - Residual Plot')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        plt.tight_layout()
        
        # Embed in Tkinter
        self.embed_plot(fig, self.tab2)
    
    def show_feature_importance(self, coefficients, title):
        """Show feature importance in the third tab"""
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': coefficients
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        
        # Embed in Tkinter
        self.embed_plot(fig, self.tab3)
    
    def show_actual_vs_predicted(self, y_true, y_pred, title):
        """Show actual vs predicted plot in the fourth tab"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred, ax=ax)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_title(f'{title} - Actual vs Predicted')
        ax.set_xlabel('Actual Prices')
        ax.set_ylabel('Predicted Prices')
        plt.tight_layout()
        
        # Embed in Tkinter
        self.embed_plot(fig, self.tab4)
    
    def embed_plot(self, fig, frame):
        """Embed a matplotlib figure in a Tkinter frame"""
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def clear_tabs(self):
        """Clear all tabs before new analysis"""
        for tab in [self.tab1, self.tab2, self.tab3, self.tab4]:
            for widget in tab.winfo_children():
                widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BostonHousingApp(root)
    root.mainloop()