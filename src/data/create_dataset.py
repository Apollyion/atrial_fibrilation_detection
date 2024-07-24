import os
from typing import List, Tuple
import wfdb
from wfdb import processing
from tqdm import tqdm
import numpy as np


def clean_annotations(annotation: wfdb.Annotation) -> Tuple[List[int], List[str]]:
    """
    Remove anotações vazias da lista de anotações.

    Args:
        annotation (wfdb.Annotation): Anotação contendo as amostras e notas auxiliares.

    Returns:
        Tuple[List[int], List[str]]: Listas de amostras e notas auxiliares não vazias.
    """
    sample, aux_note = annotation.sample, annotation.aux_note
    non_empty_indices = [i for i, note in enumerate(aux_note) if note != ""]
    clean_aux_note = [aux_note[i] for i in non_empty_indices]
    clean_sample = [sample[i] for i in non_empty_indices]
    return clean_sample, clean_aux_note


def get_ranges_afib(
    record_path: str, signal_len: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Obtém os intervalos de interesse onde a anotação é AFIB.

    Args:
      record_path (str): Caminho para o registro do sinal.
      signal_len (int): Comprimento total do sinal.

    Returns:
      Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: Uma tupla contendo duas listas de tuplas.
      A primeira lista contém os intervalos de início e fim onde a anotação é AFIB.
      A segunda lista contém os intervalos de início e fim onde a anotação é normal (não AFIB).
    """

    annotation = wfdb.rdann(record_path, "atr")
    sample, aux_note = clean_annotations(annotation)
    ranges_interest_afib = []
    ranges_interest_normal = []

    for i, label in enumerate(aux_note):
        if label == "(AFIB":
            afib_start = sample[i]
            afib_end = signal_len if i == len(sample) - 1 else sample[i + 1] - 1
            ranges_interest_afib.append((afib_start, afib_end))
        if label == "(N":
            afib_start = sample[i]
            afib_end = signal_len if i == len(sample) - 1 else sample[i + 1] - 1
            ranges_interest_normal.append((afib_start, afib_end))

    return ranges_interest_afib, ranges_interest_normal


def process_records(database_path: str, type: str, segment_size=5, base=None) -> None:
    """
    Processa todos os registros no caminho especificado, excluindo os registros indesejados.

    Args:
        database_path (str): Caminho para o diretório contendo os registros.
        type (str): Tipo de base de dados a ser processada (Classe)
        segment_size (int): Tamanho do segmento a ser extraído. Padrão: 5.
        base (str): Base de dados a ser processada. Padrão: None.
    """

    # TODO: Implementar output

    # Carregar os IDs dos registros
    record_ids = []
    with open(os.path.join(database_path, "RECORDS")) as f:
        record_ids = [line.strip() for line in f.readlines()]

    print("\n\nRecord IDs:", record_ids)

    # Remover registros indesejados da base AFDB
    if base == "afdb":
        try:
            record_ids.remove("00735")
            record_ids.remove("03665")
        except ValueError:
            pass

    # Processar cada registro
    for record_index, record_id in enumerate(record_ids):
        record_path = os.path.join(database_path, record_id)
        _, ecg_metadata = wfdb.rdsamp(record_path)
        signal_len = ecg_metadata["sig_len"]

        # Gravação completa
        recording = wfdb.rdrecord(record_name=record_path)

        if base == "afdb":
            extract_intervals_afib, extract_intervals_normal = get_ranges_afib(
                record_path, signal_len
            )
        else:
            pass

        # Para teste, processa apenas o primeiro registro
        print(
            f"Record ID: {record_id}, AFIB Intervals: {extract_intervals_afib}, Normal Intervals: {extract_intervals_normal}"
        )

        # Quantidade de RRIs em cada segmento do sinal ECG
        SEGMENT_SIZE = segment_size

        stack_rr_afib = np.empty((0, SEGMENT_SIZE), dtype=int)
        stack_recording_afib = []

        stack_rr_normal = np.empty((0, SEGMENT_SIZE), dtype=int)
        stack_recording_normal = []

        # Percorre cada trecho diagnosticado com AFIB
        for start_index, end_index in tqdm(extract_intervals_afib):
            annotations_r_peaks = wfdb.rdann(
                record_path,
                sampfrom=start_index,
                sampto=end_index,
                extension="qrs",
            )
            positions_r_peaks = annotations_r_peaks.sample
            frequency = annotations_r_peaks.fs

            positions_r_peak_ms = (positions_r_peaks / frequency) * 1000

            rr_intervals = processing.calc_rr(positions_r_peak_ms, fs=frequency)

            num_segments = (len(positions_r_peaks) - 1) // SEGMENT_SIZE

            if num_segments <= 0:
                continue

            last_segment = num_segments * SEGMENT_SIZE

            for i in range(0, last_segment, SEGMENT_SIZE):
                # Montando a saída dos segmentos (RRIs)
                rr_segment = rr_intervals[i : i + SEGMENT_SIZE]
                stack_rr_afib = np.vstack((stack_rr_afib, rr_segment))

                # Montando a saída dos segmentos (Gravação 2 derivações)
                start_index = positions_r_peaks[i]
                end_index = positions_r_peaks[i + SEGMENT_SIZE]
                rec_seg = recording.p_signal[start_index:end_index]
                stack_recording_afib.append(rec_seg)

        destination_results_afib = "afdb_results"

        if not os.path.exists(destination_results_afib):
            os.makedirs(destination_results_afib)

        np.save(
            file=f"./{destination_results_afib}/{record_index}_{record_id}_rri_segment",
            arr=stack_rr_afib,
        )

        stack_recording = np.array(stack_recording_afib, dtype=object)

        np.save(
            file=f"./{destination_results_afib}/{record_index}_{record_id}_recording_segment",
            arr=stack_recording,
        )

        # Percorre cada trecho diagnosticado como normal
        for start_index, end_index in tqdm(extract_intervals_normal):
            annotations_r_peaks = wfdb.rdann(
                record_path,
                sampfrom=start_index,
                sampto=end_index,
                extension="qrs",
            )
            positions_r_peaks = annotations_r_peaks.sample
            frequency = annotations_r_peaks.fs

            positions_r_peak_ms = (positions_r_peaks / frequency) * 1000

            rr_intervals = processing.calc_rr(positions_r_peak_ms, fs=frequency)

            num_segments = (len(positions_r_peaks) - 1) // SEGMENT_SIZE

            if num_segments <= 0:
                continue

            last_segment = num_segments * SEGMENT_SIZE

            for i in range(0, last_segment, SEGMENT_SIZE):
                # Montando a saída dos segmentos (RRIs)
                rr_segment = rr_intervals[i : i + SEGMENT_SIZE]
                stack_rr_normal = np.vstack((stack_rr_normal, rr_segment))

                # Montando a saída dos segmentos (Gravação 2 derivações)
                start_index = positions_r_peaks[i]
                end_index = positions_r_peaks[i + SEGMENT_SIZE]
                rec_seg = recording.p_signal[start_index:end_index]
                stack_recording_normal.append(rec_seg)

        destination_results_normal = "nsdb_results"

        if not os.path.exists(destination_results_normal):
            os.makedirs(destination_results_normal)

        np.save(
            file=f"./{destination_results_normal}/{record_index}_{record_id}_rri_segment",
            arr=stack_rr_normal,
        )

        stack_recording = np.array(stack_recording_normal, dtype=object)

        np.save(
            file=f"./{destination_results_normal}/{record_index}_{record_id}_recording_segment",
            arr=stack_recording,
        )

        # # Esse break serve para rodar apenas para a primeira gravação (Teste)
        break


DATABASE_PATH = "/home/apo-pc/Documents/GitHub/atrial_fibrilation_detection/data/mit-bih-atrial-fibrillation-database-1/files"
# DATABASE_PATH = "/home/apo-pc/Documents/GitHub/atrial_fibrilation_detection/data/mit-bih-normal-sinus-rhythm-database-1.0.0"
process_records(DATABASE_PATH, "AFIB", segment_size=5, base="afdb")
