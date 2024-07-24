import xmltodict
import numpy as np

for i in range(1, 4):
    # Parse the XML file
    xml_file = f"/home/flo/Documents/Projects/GCE_hist/Comparison_project/Data/diffE_isops_dNdS{i}/sources.xml"
    # Read the XML file and parse it into a dictionary
    with open(xml_file, 'r') as f:
        xml_dict = xmltodict.parse(f.read())

    # Extract sources from the dictionary
    sources = xml_dict['source_library']['source']

    # Initialize lists to store flux, ra, and dec values
    flux_values = []
    ra_values = []
    dec_values = []

    # Iterate over each source
    for source in sources:
        # Check if the source has flux, ra, and dec attributes
        if '@flux' in source and 'celestial_dir' in source['spectrum']:
            flux = float(source['@flux'])
            flux_values.append(flux)

            ra = float(source['spectrum']['celestial_dir']['@ra'])
            ra_values.append(ra)

            dec = float(source['spectrum']['celestial_dir']['@dec'])
            dec_values.append(dec)

    # Convert lists to numpy arrays
    flux_array = np.array(flux_values)
    ra_array = np.array(ra_values)
    dec_array = np.array(dec_values)

    # Print the numpy arrays
    print("Flux array:", flux_array)
    print("RA array:", ra_array)
    print("Dec array:", dec_array)

    # Make a scatter plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(ra_array, dec_array, c=flux_array, s=100, cmap='viridis')

    # Store as a npz file
    np.savez(f"/home/flo/Documents/Projects/GCE_hist/Comparison_project/Data/diffE_isops_dNdS{i}/source_info.npz", flux=flux_array, ra=ra_array, dec=dec_array)
