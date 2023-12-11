def map_positions(positions, patient_ids):
    # Verifica che le lunghezze dei due array siano uguali
    if len(positions) != len(patient_ids):
        raise ValueError("Le lunghezze degli array non corrispondono.")

    # Crea un array che contiene gli ID pazienti nell'ordine specificato dalle posizioni
    ordered_patient_ids = [patient_ids[position - 1] for position in positions]

    return ordered_patient_ids

""" positions_array = [0, 2, 1, 5, 1]
patient_ids_array = ['A', 'B', 'C', 'D', 'E']

ordered_ids_result = map_positions(positions_array, patient_ids_array)
print(ordered_ids_result) """
