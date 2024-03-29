# Calculation placed in Supplementary Informations
def tensor_rotation(eo, psi=30, theta=90, phi=150):
    import numpy as np
#     eo = np.array([[0, 0, 0, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0], [0.13, 0.13, 0.8, 0, 0, 0]])

    # Compute the elements of matrix A
    A = np.array([
    [np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi), np.cos(phi) * np.sin(psi) + np.cos(theta) * np.cos(psi) * np.sin(phi), np.sin(theta) * np.sin(phi)],
    [-np.cos(theta) * np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi), np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi), np.cos(theta) * np.sin(psi)],
    [np.sin(theta) * np.sin(psi), -np.cos(phi) * np.sin(theta), np.cos(theta)]
    ])

    # Compute the elements of matrix N
    N = np.array([
        [A[0, 0] ** 2, A[1, 0] ** 2, A[2, 0] ** 2, 2 * A[1, 0] * A[2, 0], 2 * A[2, 0] * A[0, 0], 2 * A[0, 0] * A[1, 0]],
        [A[0, 1] ** 2, A[1, 1] ** 2, A[2, 1] ** 2, 2 * A[1, 1] * A[2, 1], 2 * A[2, 1] * A[0, 1], 2 * A[0, 1] * A[1, 1]],
        [A[0, 2] ** 2, A[1, 2] ** 2, A[2, 2] ** 2, 2 * A[1, 2] * A[2, 2], 2 * A[2, 2] * A[0, 2], 2 * A[0, 2] * A[1, 2]],
        [A[0, 1] * A[0, 2], A[1, 1] * A[1, 2], A[2, 1] * A[2, 2], A[1, 1] * A[2, 2] + A[2, 1] * A[1, 2], A[0, 1] * A[2, 2] + A[2, 1] * A[0, 2], A[1, 1] * A[0, 2] + A[0, 1] * A[1, 2]],
        [A[0, 2] * A[0, 0], A[1, 2] * A[1, 0], A[2, 2] * A[2, 0], A[1, 2] * A[2, 0] + A[2, 2] * A[1, 0], A[2, 2] * A[0, 0] + A[0, 2] * A[2, 0], A[0, 2] * A[1, 0] + A[0, 0] * A[1, 2]],
        [A[0, 0] * A[0, 1], A[1, 0] * A[1, 1], A[2, 0] * A[2, 1], A[1, 0] * A[2, 1] + A[2, 0] * A[1, 1], A[2, 0] * A[0, 1] + A[0, 0] * A[2, 1], A[0, 0] * A[1, 1] + A[1, 0] * A[0, 1]]
    ])

    # Initialize e' matrix
    e_prime = np.zeros((3, 6))

    # Calculate each element of e' matrix
    for i in range(3):
        for j in range(6):
            for k in range(3):
                for l in range(6):
                    e_prime[i, j] += A[i, k] * eo[k, l] * N[l, j]

    # Print the resulting e' matrix
#     print(e_prime)
    return e_prime
    
    
    ###########################################################################################################################################3
def tensor_rotation_plot(eo, phi = 30,order=[0,0]):
    import numpy as np
    import plotly.graph_objects as go 
    
    # Define the angles phi, theta, and psi
#     phi_vals = 30
    theta_vals = np.linspace(0, np.pi, 40)
    psi_vals = np.linspace(0, 2 * np.pi, 40)
    

    # Initialize array to store the e'11 matrix elements for each combination of psi, theta, and phi
    e_prime_11 = np.zeros((len(psi_vals), len(theta_vals)))

    # Calculate the e'11 matrix elements for each combination of psi, theta, and phi
    for i, psi in enumerate(psi_vals):
        for j, theta in enumerate(theta_vals):
            # Compute the elements of matrix A
            A = np.array([
            [np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi), np.cos(phi) * np.sin(psi) + np.cos(theta) * np.cos(psi) * np.sin(phi), np.sin(theta) * np.sin(phi)],
            [-np.cos(theta) * np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi), np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi), np.cos(theta) * np.sin(psi)],
            [np.sin(theta) * np.sin(psi), -np.cos(phi) * np.sin(theta), np.cos(theta)]
            ])
            
            # Compute the elements of matrix N
            N = np.array([
                [A[0, 0]**2, A[1, 0]**2, A[2, 0]**2, 2*A[1, 0]*A[2, 0], 2*A[2, 0]*A[0, 0], 2*A[0, 0]*A[1, 0]],
                [A[0, 1]**2, A[1, 1]**2, A[2, 1]**2, 2*A[1, 1]*A[2, 1], 2*A[2, 1]*A[0, 1], 2*A[0, 1]*A[1, 1]],
                [A[0, 2]**2, A[1, 2]**2, A[2, 2]**2, 2*A[1, 2]*A[2, 2], 2*A[2, 2]*A[0, 2], 2*A[0, 2]*A[1, 2]],
                [A[0, 1]*A[0, 2], A[1, 1]*A[1, 2], A[2, 1]*A[2, 2], A[1, 1]*A[2, 2]+A[2, 1]*A[1, 2], A[0, 1]*A[2, 2]+A[2, 1]*A[0, 2], A[1, 1]*A[0, 2]+A[0, 1]*A[1, 2]],
                [A[0, 2]*A[0, 0], A[1, 2]*A[1, 0], A[2, 2]*A[2, 0], A[1, 2]*A[2, 0]+A[2, 2]*A[1, 0], A[2, 2]*A[0, 0]+A[0, 2]*A[2, 0], A[0, 2]*A[1, 0]+A[0, 0]*A[1, 2]],
                [A[0, 0]*A[0, 1], A[1, 0]*A[1, 1], A[2, 0]*A[2, 1], A[1, 0]*A[2, 1]+A[2, 0]*A[1, 1], A[2, 0]*A[0, 1]+A[0, 0]*A[2, 1], A[0, 0]*A[1, 1]+A[1, 0]*A[0, 1]]
            ])

            # Compute the elements of the e' matrix
            e_prime = np.zeros((3, 6))
            for l in range(3):
                for m in range(6):
                    for n in range(3):
                        for o in range(6):
                            e_prime[l, m] += A[l, n]*eo[n,o] * N[o, n]

            # Store the e'11 matrix element at the corresponding indices
            e_prime_11[i, j] = e_prime[order[0], order[1]]

    # Convert angles to degrees
    theta_vals_deg = np.degrees(theta_vals)
    psi_vals_deg = np.degrees(psi_vals)

    # Find the maximum point and its value

    # Create spherical plot
    fig = go.Figure(data=[go.Surface(
        x=theta_vals_deg,
        y=psi_vals_deg,
        z=e_prime_11,
        colorscale='Jet',
        showscale=False,
        hovertemplate = 'Theta : %{x:.2f}'+\
                '<br>Psi : %{y:.2f}'+\
                '<br>e_r : %{z:.3f}<extra></extra>'
    )])
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
#         title=r'Spherical Plot of e\'_11',
    #     plot_bgcolor='white',  # Set background color to white
        scene=dict(
            xaxis_title='Theta',
            yaxis_title='Psi',
            zaxis_title=r"e'"+str(order[0]+1)+str(order[1]+1),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=16, color='black'),  # Set tick label color to black
                title_font=dict(size=20, color='red'),  # Set axis label color to black
#                 showline=True,
#                 linecolor='black',  # Set axis line color to black
                tickangle=0,  # Rotate x-axis tick labels
                nticks=5,  # Set the maximum number of ticks
                tick0=0,  # Set the starting tick
                dtick=60,  # Set the tick interval
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=16, color='black'),  # Set tick label color to black
                title_font=dict(size=20, color='red'),  # Set axis label color to black
#                 showline=True,
#                 linecolor='black',  # Set axis line color to black
                tickangle=-5,  # Rotate x-axis tick labels
                nticks=5,  # Set the maximum number of ticks
                tick0=0,  # Set the starting tick
                dtick=90,  # Set the tick interval
            ),
            zaxis=dict(
                showgrid=False,
                tickfont=dict(size=16, color='black'),  # Set tick label color to black
                title_font=dict(size=20, color='red'),  # Set axis label color to black
#                 showline=True,
                linecolor='black',  # Set axis line color to black
                tickangle=-0,
                ticks = "outside", tickcolor='white', ticklen=10,
#                 ticklen=20,
                
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.05, z=1.25),  # Adjust camera position
                projection=dict(type='orthographic'),  # Change projection type
            )
        ),


        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
        paper_bgcolor='rgba(255, 255, 255, 1)',  # Set paper background color to white
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Set plot background color to white
    #     bgcolor='rgba(255, 255, 255, 1)',  # Set layout background color to white

    )

    fig.show()
    
    # Annotation of Max Value
    max_index = np.unravel_index(np.argmax(e_prime_11), e_prime_11.shape)
    max_theta = theta_vals_deg[max_index[1]]
    max_psi = psi_vals_deg[max_index[0]]
    max_e11 = e_prime_11[max_index]

    print(f"Maximum Value: {max_e11:.2f}")
    print(f"Theta: {max_theta:.2f}")
    print(f"Psi: {max_psi:.2f}")
    print(f"Phi: {phi:.0f}")
    
    return fig, max_e11, max_theta, max_psi, phi


#####################################################################################################################################
########################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

def tensor_rotation_optimization(eo, order=[0, 0]):
    # Define the angles theta and psi
    theta_vals = np.linspace(0, np.pi, 50)
    psi_vals = np.linspace(0, 2 * np.pi, 50)
    phi_vals = np.linspace(0, 2 * np.pi, 50)

    # Initialize arrays to store the maximum values
    max_e11_vals = []
    max_theta_vals = []
    max_psi_vals = []

    # Iterate over each phi value
    for phi in phi_vals:
        # Initialize array to store the e'11 matrix elements for each combination of psi, theta, and phi
        e_prime_11 = np.zeros((len(psi_vals), len(theta_vals)))

        # Calculate the e'11 matrix elements for each combination of psi, theta, and phi
        for i, psi in enumerate(psi_vals):
            for j, theta in enumerate(theta_vals):
                # Compute the elements of matrix A
                A = np.array([
                    [np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                     np.cos(phi) * np.sin(psi) + np.cos(theta) * np.cos(psi) * np.sin(phi),
                     np.sin(theta) * np.sin(phi)],
                    [-np.cos(theta) * np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi),
                     np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                     np.cos(theta) * np.sin(psi)],
                    [np.sin(theta) * np.sin(psi), -np.cos(phi) * np.sin(theta), np.cos(theta)]
                ])

                # Compute the elements of matrix N
                N = np.array([
                    [A[0, 0]**2, A[1, 0]**2, A[2, 0]**2, 2 * A[1, 0] * A[2, 0], 2 * A[2, 0] * A[0, 0], 2 * A[0, 0] * A[1, 0]],
                    [A[0, 1]**2, A[1, 1]**2, A[2, 1]**2, 2 * A[1, 1] * A[2, 1], 2 * A[2, 1] * A[0, 1], 2 * A[0, 1] * A[1, 1]],
                    [A[0, 2]**2, A[1, 2]**2, A[2, 2]**2, 2 * A[1, 2] * A[2, 2], 2 * A[2, 2] * A[0, 2], 2 * A[0, 2] * A[1, 2]],
                    [A[0, 1] * A[0, 2], A[1, 1] * A[1, 2], A[2, 1] * A[2, 2], A[1, 1] * A[2, 2] + A[2, 1] * A[1, 2], A[0, 1] * A[2, 2] + A[2, 1] * A[0, 2], A[1, 1] * A[0, 2] + A[0, 1] * A[1, 2]],
                    [A[0, 2] * A[0, 0], A[1, 2] * A[1, 0], A[2, 2] * A[2, 0], A[1, 2] * A[2, 0] + A[2, 2] * A[1, 0], A[2, 2] * A[0, 0] + A[0, 2] * A[2, 0], A[0, 2] * A[1, 0] + A[0, 0] * A[1, 2]],
                    [A[0, 0] * A[0, 1], A[1, 0] * A[1, 1], A[2, 0] * A[2, 1], A[1, 0] * A[2, 1] + A[2, 0] * A[1, 1], A[2, 0] * A[0, 1] + A[0, 0] * A[2, 1], A[0, 0] * A[1, 1] + A[1, 0] * A[0, 1]]
                ])

                # Compute the elements of the e' matrix
                e_prime = np.zeros((3, 6))
                for l in range(3):
                    for m in range(6):
                        for n in range(3):
                            for o in range(6):
                                e_prime[l, m] += A[l, n] * eo[n, o] * N[o, n]

                # Store the e'11 matrix element at the corresponding indices
                e_prime_11[i, j] = e_prime[order[0], order[1]]

        # Find the maximum point and its value for each order
        max_index = np.unravel_index(np.argmax(e_prime_11), e_prime_11.shape)
        max_theta = np.degrees(theta_vals[max_index[1]])
        max_psi = np.degrees(psi_vals[max_index[0]])
        max_e11 = e_prime_11[max_index]

        # Append the maximum values to the respective arrays
        max_e11_vals.append(max_e11)
        max_theta_vals.append(max_theta)
        max_psi_vals.append(max_psi)

    # Plot the maximum values for each order as a function of phi
    
    plt.figure(figsize=(6, 4))
    plt.plot(phi_vals, max_e11_vals,color='blue')
    plt.xlabel('Phi (degrees)', fontsize=20, color='red')  # Increase font size of xlabel
    plt.ylabel("e' " + str(order[0]+1) + str(order[1]+1)+r"  $C/m^2$", fontsize=20, color='red')  # Increase font size of ylabel
    plt.tick_params(axis='both', which='major', labelsize=14)  # Increase font size of tick labels
    
    # Convert x-ticks to degrees and show only 8 ticks without decimal places
    xtick_vals = np.linspace(np.degrees(phi_vals[0]), np.degrees(phi_vals[-1]), 7)
    xtick_vals = [int(x) for x in xtick_vals]
    plt.xticks(np.radians(xtick_vals), xtick_vals)

    # Annotate the maximum value
    max_phi_index = np.argmax(max_e11_vals)  # Index of the maximum value
    max_phi = phi_vals[max_phi_index]  # Corresponding phi value
    max_e11 = max_e11_vals[max_phi_index]  # Maximum e'11 value
    
    

    plt.annotate(f'Max: {max_e11:.2f} @ Angle: {60*max_phi:.2f}°', xy=(max_phi, max_e11),
            xytext=(0.08, 1.05), textcoords='axes fraction',
                 fontname = 'serif',
                 fontsize=12,
                 color='k',
                )
    
    # Add dashed vertical line from maximum value to y_max
    plt.axvline(x=max_phi, ymin=0, ymax=max_e11 / plt.ylim()[1], color='gray',alpha=0.25, linestyle='--')
    plt.axhline(y=max_e11, xmin=0, xmax=max_phi / plt.xlim()[1], color='gray', alpha=0.25, linestyle='--')
    plt.show()


    return plt
