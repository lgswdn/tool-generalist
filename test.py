from isaaclab.app import AppLauncher

# Initialize the simulation app
app_launcher = AppLauncher(
    headless=True, 
    kit_args="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"
)
simulation_app = app_launcher.app

print("Initialization complete.")

# Close immediately
simulation_app.close()