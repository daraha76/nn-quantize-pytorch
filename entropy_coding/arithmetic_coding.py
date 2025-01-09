from decimal import *
import math
import random
import time


class ArithmeticCoder:
    """
    ArithmeticCoder is a class for building the arithmetic encoding.
    """
    
    # TODO list
    # [V] 0. modify precision of Decimal type
    # [V] 1. freq_table and prob_table into list, now assume input symbols are integer
    # [V] 2. set boundary of codable input range -> convert into freq_table index
    # [V] 3. list of freq_table (freq_table is changed for each input symbol)

    def __init__(self, frequency_table, term_list, term_type, decimal_precision=32, table_type='fixed', max_msg_len=None, include_oob=False, save_stages=False):
        """
        frequency_table: Frequency of each term described by the term_list. zero probability is allowed. List of list if table_type == 'dynamic'
        term_list: {fixed} (str) list of terms, must be non overlapped! (int) start and end point of continuous integer sequence
                   {dynamic} (str) list of ... (int) list of ...
        term_type: define type of the terms
        decimal_precision: precision of the Decimal type
        table_type: (fixed) use same prob_table for whole message, (dynamic) prob_table changes for each term in message, message length must be defined! (adaptive) TODO
        max_msg_len: maximum length of input message, used only when table_type == 'dynamic'
        include_oob: if True, (int) additional two terms in the frequency table denotes the freq. of low_oob, high_oob.
        save_stages: If True, then the intervals of each stage are saved in a list. Note that setting save_stages=True may cause memory overflow if the message is large
        """
        getcontext().prec = decimal_precision
        self.save_stages = save_stages
        if(save_stages == True):
            print("WARNING: Setting save_stages=True may cause memory overflow if the message is large.")

        self.include_oob = include_oob
        assert term_type in ['int', 'str']
        self.term_type = term_type
        self.term_list = term_list
        
        assert table_type in ['fixed', 'dynamic']
        self.table_type = table_type
        self.max_msg_len = max_msg_len
        if self.table_type == 'dynamic':
            assert len(frequency_table) == self.max_msg_len
        self.num_term = self.get_num_terms()
        self.probability_table = self.get_probability_table(frequency_table)
    
    def get_num_terms(self):
        """
        Calculates the number of terms.
        """
        if self.table_type == 'fixed':
            if self.term_type == 'str':
                num_term = len(self.term_list)
            elif self.term_type == 'int':
                num_term = self.term_list[1] - self.term_list[0] + 1
            if self.include_oob:
                num_term += 2
                
        elif self.table_type == 'dynamic':
            num_term = []
            for msg_idx in range(self.max_msg_len):
                tmp_num_term = 0
                if self.term_type == 'str':
                    tmp_num_term = len(self.term_list[msg_idx])
                elif self.term_type == 'int':
                    tmp_num_term = int(self.term_list[msg_idx][1]) - int(self.term_list[msg_idx][0]) + 1
                if self.include_oob:
                    tmp_num_term += 2
                num_term.append(tmp_num_term)
        
        return num_term
        
    def get_probability_table(self, frequency_table):
        """
        Calculates the probability table out of the frequency table.

        frequency_table: A table of the term frequencies.

        Returns the probability table. (sum=1)
        """
        if self.table_type == 'fixed':
            total_frequency = sum(frequency_table[:self.num_term])
            probability_table = [Decimal(term_frequency / total_frequency) for term_frequency in frequency_table[:self.num_term]]   # Convert into Decimal type

        elif self.table_type == 'dynamic':
            probability_table = []
            for msg_idx, tmp_table in enumerate(frequency_table):
                tmp_table = tmp_table[:self.num_term[msg_idx]]
                total_frequency = Decimal('0.0')
                for tmp_term in tmp_table:
                    total_frequency += Decimal(tmp_term)
                probability_table.append([Decimal(term_frequency) / total_frequency for term_frequency in tmp_table])
                
                # total_frequency = sum(tmp_table)
                # probability_table.append([Decimal(term_frequency / total_frequency) for term_frequency in tmp_table])   # Convert into Decimal type
                
            for tb_idx, tmp_table in enumerate(probability_table):
                tmp_sum = Decimal('0.0')
                for tmp_term in tmp_table:
                    tmp_sum += tmp_term
                if tmp_sum > Decimal('1.0'):
                    print(f"\nSum over 1.0 at {tb_idx + 1}th table!")
                
        return probability_table
    
    def calculate_entropy(self):
        """
        Calcualte entropy from the probability table
        """
        entropy = 0.0
        if self.table_type == 'fixed':
            for term_prob in self.probability_table:
                term_prob = float(term_prob)
                if term_prob != 0:
                    entropy += term_prob * (-math.log2(term_prob))
                    
        if self.table_type == 'dynamic':
            for tmp_table in self.probability_table:
                for term_prob in tmp_table:
                    term_prob = float(term_prob)
                    if term_prob != 0:
                        entropy += term_prob * (-math.log2(term_prob))
            entropy /= len(self.max_msg_len)
        
        return entropy
    
    def calculate_optimal_codelength(self, msg):
        """
        Calcualte theoretical minimum of codelength
        """
        if isinstance(msg, str):
            msg = list(msg)

        msg = self.term_to_idx(msg)     # From now on, msg contains list of indices for prob_table
        
        codelength = 0.0
        for msg_idx, term in enumerate(msg):
            if self.table_type == 'fixed':
                codelength += -math.log2(self.probability_table[term])
            elif self.table_type == 'dynamic':
                codelength += -math.log2(self.probability_table[msg_idx][term])
        
        return codelength

    def term_to_idx(self, msg_in_term):
        """
        Convert list of terms into list of  prob_table indices
        msg_in_term: list of terms
        """
        msg_in_idx = []
        if self.term_type == 'str':
            for m_idx, term in enumerate(msg_in_term):
                if self.table_type == 'fixed':
                    msg_in_idx.append(self.term_list.index(term))
                elif self.table_type == 'dynamic':
                    msg_in_idx.append(self.term_list[m_idx].index(term))
        
        elif self.term_type == 'int':
            for m_idx, term in enumerate(msg_in_term):
                if self.table_type == 'fixed':
                    msg_in_idx.append(int(term - self.term_list[0]))
                elif self.table_type == 'dynamic':
                    msg_in_idx.append(int(term - self.term_list[m_idx][0]))
        return msg_in_idx
    
    def idx_to_term(self, msg_in_idx):
        """
        Convert list of prob_table indices into list of terms
        msg_in_term: list of prob_table indices
        """
        msg_in_term = []
        if self.term_type == 'str':
            for m_idx, idx in enumerate(msg_in_idx):
                if self.table_type == 'fixed':
                    msg_in_term.append(self.term_list[idx])
                elif self.table_type == 'dynamic':
                    msg_in_term.append(self.term_list[m_idx][idx])
        elif self.term_type == 'int':
             for m_idx, idx in enumerate(msg_in_idx):
                if self.table_type == 'fixed':
                    msg_in_term.append(idx + self.term_list[0])
                elif self.table_type == 'dynamic':
                    msg_in_term.append(idx + self.term_list[m_idx][0])
        return msg_in_term

    def get_encoded_value(self, last_stage_probs):
        """
        Depricated!!
        After encoding the entire message, this method returns the single value that represents the entire message.

        last_stage_probs: A list of the probabilities in the last stage.
        
        Returns the minimum and maximum probabilites in the last stage in addition to the value encoding the message.
        """
        # last_stage_probs = list(last_stage_probs.values())
        # last_stage_values = []
        # for sublist in last_stage_probs:
        #     for element in sublist:
        #         last_stage_values.append(element)

        # last_stage_min = min(last_stage_values)
        # last_stage_max = max(last_stage_values)
        # encoded_value = (last_stage_min + last_stage_max)/2
        
        last_stage_min = last_stage_probs[0][0]
        last_stage_max = last_stage_probs[-1][1]
        encoded_value = (last_stage_min + last_stage_max)/2

        return last_stage_min, last_stage_max, encoded_value

    def process_stage(self, stage_min, stage_max, msg_idx=None):
        """
        Processing a stage in the encoding/decoding process.

        stage_min: The minumim probability of the current stage.
        stage_max: The maximum probability of the current stage.
        
        Returns the boundary probabilities for each term in the stage.
        """
        # stage_probs = {}
        # stage_domain = stage_max - stage_min
        # for term_idx in range(len(probability_table.items())):
        #     term = list(probability_table.keys())[term_idx]
        #     term_prob = Decimal(probability_table[term])
        #     cum_prob = term_prob * stage_domain + stage_min
        #     stage_probs[term] = [stage_min, cum_prob]
        #     stage_min = cum_prob
            
        stage_probs = []
        cum_prob = stage_min
        stage_domain = stage_max - stage_min
        if self.table_type == 'fixed':
            prob_table = self.probability_table
        elif self.table_type == 'dynamic':
            prob_table = self.probability_table[msg_idx]
        
        for term_prob in prob_table:
            cum_tmp = cum_prob + term_prob * stage_domain
            stage_probs.append(cum_prob)
            cum_prob = cum_tmp
            # Bound cumulateve prob to prevent errors
            # if cum_tmp <= stage_max:
            #     cum_prob = cum_tmp
            # else:
            #     cum_prob = stage_max
        stage_probs.append(stage_max)
        
        return stage_probs

    def encode(self, msg):
        """
        Encodes a message using arithmetic encoding.

        msg: The message to be encoded.

        Returns the encoder, the floating-point value representing the encoded message, and the maximum and minimum values of the interval in which the floating-point value falls.
        """
        
        if isinstance(msg, str):
            msg = list(msg)
        
        if self.table_type == 'dynamic':
            assert len(msg) <= self.max_msg_len

        msg = self.term_to_idx(msg)     # From now on, msg contains list of indices for prob_table

        encoder = []

        stage_min = Decimal('0.0')
        stage_max = Decimal('1.0')

        for msg_idx, term in enumerate(msg):
            #print(f"#Terms: {len(self.probability_table[msg_idx])}, term_list: {self.term_list[msg_idx]} current term: {term}")
            stage_probs = self.process_stage(stage_min, stage_max, msg_idx)

            stage_min, stage_max = stage_probs[term: term + 2]

            if self.save_stages:
                encoder.append(stage_probs)
            
            # if stage_max > Decimal('1.0'):
            #     print()
            #     print(stage_min)
            #     print(stage_max)
            #     print(stage_probs)

        interval_min_value, interval_max_value = stage_min, stage_max
        encoded_msg = (interval_min_value + interval_max_value) / 2
        
        # last_stage_probs = self.process_stage(stage_min, stage_max)
        
        # if self.save_stages:
        #     encoder.append(last_stage_probs)

        # interval_min_value, interval_max_value, encoded_msg = self.get_encoded_value(last_stage_probs)

        return encoded_msg, encoder, interval_min_value, interval_max_value

    def process_stage_binary(self, float_interval_min, float_interval_max, stage_min_bin, stage_max_bin):
        """
        Processing a stage in the encoding/decoding process.

        float_interval_min: The minimum floating-point value in the interval in which the floating-point value that encodes the message is located.
        float_interval_max: The maximum floating-point value in the interval in which the floating-point value that encodes the message is located.
        stage_min_bin: The minimum binary number in the current stage.
        stage_max_bin: The maximum binary number in the current stage.

        Returns the probabilities of the terms in this stage. There are only 2 terms.
        """

        stage_mid_bin = stage_min_bin + "1"
        stage_min_bin = stage_min_bin + "0"

        stage_probs = {}
        stage_probs[0] = [stage_min_bin, stage_mid_bin]
        stage_probs[1] = [stage_mid_bin, stage_max_bin]

        return stage_probs

    def encode_binary(self, float_interval_min, float_interval_max, num_max_trial=256):
        """
        Calculates the binary code that represents the floating-point value that encodes the message.

        float_interval_min: The minimum floating-point value in the interval in which the floating-point value that encodes the message is located.
        float_interval_max: The maximum floating-point value in the interval in which the floating-point value that encodes the message is located.

        Returns the binary code representing the encoded message.
        """

        binary_encoder = []
        binary_code = None

        stage_min_bin = "0.0"
        stage_max_bin = "1.0"

        stage_probs = {}
        stage_probs[0] = [stage_min_bin, "0.1"]
        stage_probs[1] = ["0.1", stage_max_bin]
        
        num_trial = 0
        failed = False
        while True:
            num_trial += 1
            
            if float_interval_max < bin2float(stage_probs[0][1]):
                stage_min_bin = stage_probs[0][0]
                stage_max_bin = stage_probs[0][1]
            else:
                stage_min_bin = stage_probs[1][0]
                stage_max_bin = stage_probs[1][1]

            if self.save_stages:
                binary_encoder.append(stage_probs)

            stage_probs = self.process_stage_binary(float_interval_min,
                                                    float_interval_max,
                                                    stage_min_bin,
                                                    stage_max_bin)

            # print(stage_probs[0][0], bin2float(stage_probs[0][0]))
            # print(stage_probs[0][1], bin2float(stage_probs[0][1]))
            if (bin2float(stage_probs[0][0]) >= float_interval_min) and (bin2float(stage_probs[0][1]) < float_interval_max):
                # The binary code is found.
                # print(stage_probs[0][0], bin2float(stage_probs[0][0]))
                # print(stage_probs[0][1], bin2float(stage_probs[0][1]))
                # print("The binary code is : ", stage_probs[0][0])
                binary_code = stage_probs[0][0]
                break
            elif (bin2float(stage_probs[1][0]) >= float_interval_min) and (bin2float(stage_probs[1][1]) < float_interval_max):
                # The binary code is found.
                # print(stage_probs[1][0], bin2float(stage_probs[1][0]))
                # print(stage_probs[1][1], bin2float(stage_probs[1][1]))
                # print("The binary code is : ", stage_probs[1][0])
                binary_code = stage_probs[1][0]
                break
        
            # TODO...
            if num_trial >= num_max_trial:
                binary_code = stage_probs[1][0]
                failed = True
                break
                

        if self.save_stages:
            binary_encoder.append(stage_probs)

        return binary_code, binary_encoder, failed 

    def decode(self, encoded_msg, msg_length):
        """
        Decodes a message from a floating-point number.
        
        encoded_msg: The floating-point value that encodes the message.
        msg_length: Length of the message.
        probability_table: The probability table.

        Returns the decoded message.
        """

        decoder = []

        decoded_msg = []    # Currently, list of prob_table indices

        stage_min = Decimal('0.0')
        stage_max = Decimal('1.0')

        for msg_idx in range(msg_length):
            stage_probs = self.process_stage(stage_min, stage_max, msg_idx)

            for t_idx, term_max in enumerate(stage_probs[1:]):
                if encoded_msg <= term_max:
                    decoded_msg.append(t_idx)
                    stage_min, stage_max = stage_probs[t_idx: t_idx + 2]
                    break
                elif t_idx == len(stage_probs) - 2:
                    1

            if self.save_stages:
                decoder.append(stage_probs)

        # if self.save_stages:
        #     last_stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        #     decoder.append(last_stage_probs)
        
        decoded_msg = self.idx_to_term(decoded_msg)

        return decoded_msg, decoder

def float2bin(float_num, num_bits=None):
    """
    Converts a floating-point number into binary.

    float_num: The floating-point number. 
    num_bits: The number of bits expected in the result. If None, then the number of bits depends on the number.

    Returns the binary representation of the number.
    """

    float_num = str(float_num)
    if float_num.find(".") == -1:
        # No decimals in the floating-point number.
        integers = float_num
        decimals = ""
    else:
        integers, decimals = float_num.split(".")
    decimals = "0." + decimals
    decimals = Decimal(decimals)
    integers = int(integers)

    result = ""
    num_used_bits = 0
    while True:
        mul = decimals * 2
        int_part = int(mul)
        result = result + str(int_part)
        num_used_bits = num_used_bits + 1

        decimals = mul - int(mul)
        if type(num_bits) is type(None):
            if decimals == 0:
                break
        elif num_used_bits >= num_bits:
            break
    if type(num_bits) is type(None):
        pass
    elif len(result) < num_bits:
        num_remaining_bits = num_bits - len(result)
        result = result + "0"*num_remaining_bits

    integers_bin = bin(integers)[2:]
    result = str(integers_bin) + "." + str(result)
    return result

def bin2float(bin_num):
    """
    Converts a binary number to a floating-point number.

    bin_num: The binary number as a string.

    Returns the floating-point representation.
    """

    if bin_num.find(".") == -1:
        # No decimals in the binary number.
        integers = bin_num
        decimals = ""
    else:
        integers, decimals = bin_num.split(".")
    result = Decimal(0.0)

    # Working with integers.
    for idx, bit in enumerate(integers):
        if bit == "0":
            continue
        mul = 2**idx
        result = result + Decimal(mul)

    # Working with decimals.
    for idx, bit in enumerate(decimals):
        if bit == "0":
            continue
        mul = Decimal(1.0)/Decimal((2**(idx+1)))
        result = result + mul
    return result


if __name__ == "__main__":

    #term_type = 'str'
    #term_list = ['q', 'w', 'e', 'r', 't']
    #frequency_table = [13, 24, 34, 3, 8]
    
    #term_type = 'int'
    #term_list = [1, 10]
    #frequency_table = [1, 5, 11, 17, 29, 41, 40, 25, 13, 4]
    
    term_type = 'int'
    table_type = 'dynamic'
    max_msg_len = 32
    term_list = []
    frequency_table = []
    for m_idx in range(max_msg_len):
        term_min = 0
        term_max = random.randint(5, 33)
        term_list.append([term_min, term_max])
        frequency_table.append([random.randint(1, 100) for _ in range(term_max - term_min + 1)])

    AE = ArithmeticCoder(frequency_table, term_list, term_type, table_type=table_type, max_msg_len=max_msg_len, decimal_precision=1024)

    #input_msg = "qeewtwwqwwere"
    #input_msg = [term_list[random.randrange(0, len(term_list))] for _ in range(32)]
    #input_msg = [random.randrange(term_list[0], term_list[1] + 1) for _ in range(32)]
    input_msg = []
    for m_idx in range(max_msg_len):
        input_msg.append(random.randint(term_list[m_idx][0], term_list[m_idx][1]))

    time_start = time.time()
    encoded_msg, _, interval_min_value, interval_max_value = AE.encode(input_msg)
    time_end = time.time()
    time_encode = time_end - time_start
    
    time_start = time.time()
    binary_code, _ = AE.encode_binary(float_interval_min=interval_min_value,
                                               float_interval_max=interval_max_value)
    time_end = time.time()
    time_binary = time_end - time_start
    
    time_start = time.time()
    decoded_msg, _ = AE.decode(encoded_msg, len(input_msg))
    time_end = time.time()
    time_decode = time_end - time_start
    
    print(f"time elapsed: {time_encode + time_binary + time_decode:.4f} sec  |  {time_encode:.4f}  |  {time_binary:.4f}  |  {time_decode:.4f}")
    #print(input_msg)
    #print(binary_code)
    #print(decoded_msg)
    
    msg_error = [msg_i - msg_o for msg_i, msg_o in zip(input_msg, decoded_msg)]
    #print(msg_error)
    #print(sum(msg_error))
    
    optim_codelen = AE.calculate_optimal_codelength(input_msg)
    print(optim_codelen)
    print(len(binary_code)-2)
    print(f"Overhead: {(len(binary_code) - 2 - optim_codelen) / optim_codelen * 100} %")