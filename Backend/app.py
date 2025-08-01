import gradio as gr
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

class IntrusionClassifier(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.linear1 = nn.Linear(117, 64)
        self.linear2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xb):
        xb = self.linear1(xb)
        xb = self.relu(xb)
        xb = self.dropout(xb)
        xb = self.linear2(xb)
        return xb

model = IntrusionClassifier(0.5)
model.load_state_dict(torch.load('intrusion_model.pth', map_location=torch.device('cpu')))
model.eval()

preprocessor = joblib.load('preprocessor.joblib')
inverse_class_map = {0: 'anomaly', 1: 'normal'}

feature_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

protocol_types = ['tcp', 'udp', 'icmp']
services = ['ftp_data', 'other', 'private', 'http', 'remote_job', 'name', 'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u', 'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp', 'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap', 'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois', 'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login', 'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u', 'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell', 'netstat', 'pop_3', 'nnsp', 'pop_2', 'printer', 'tim_i', 'rje', 'red_i', 'netbios_ssn', 'tftp_u', 'X11', 'IRC']
flags = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH']


def predict(*args):
    input_df = pd.DataFrame([args], columns=feature_columns)
    processed_input = preprocessor.transform(input_df)
    input_tensor = torch.tensor(processed_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        prediction_label = inverse_class_map[predicted_idx.item()]


    return prediction_label.upper()


inputs = [
    gr.Number(label="duration", value=0),
    gr.Dropdown(choices=protocol_types, label="protocol_type", value="tcp"),
    gr.Dropdown(choices=services, label="service", value="http"),
    gr.Dropdown(choices=flags, label="flag", value="SF"),
    gr.Number(label="src_bytes", value=0),
    gr.Number(label="dst_bytes", value=0),
    gr.Radio(choices=[0, 1], label="land", value=0),
    gr.Number(label="wrong_fragment", value=0),
    gr.Number(label="urgent", value=0),
    gr.Number(label="hot", value=0),
    gr.Number(label="num_failed_logins", value=0),
    gr.Radio(choices=[0, 1], label="logged_in", value=0),
    gr.Number(label="num_compromised", value=0),
    gr.Radio(choices=[0, 1], label="root_shell", value=0),
    gr.Number(label="su_attempted", value=0),
    gr.Number(label="num_root", value=0),
    gr.Number(label="num_file_creations", value=0),
    gr.Number(label="num_shells", value=0),
    gr.Number(label="num_access_files", value=0),
    gr.Number(label="num_outbound_cmds", value=0),
    gr.Radio(choices=[0, 1], label="is_host_login", value=0),
    gr.Radio(choices=[0, 1], label="is_guest_login", value=0),
    gr.Number(label="count", value=0),
    gr.Number(label="srv_count", value=0),
    gr.Slider(0, 1, label="serror_rate", value=0),
    gr.Slider(0, 1, label="srv_serror_rate", value=0),
    gr.Slider(0, 1, label="rerror_rate", value=0),
    gr.Slider(0, 1, label="srv_rerror_rate", value=0),
    gr.Slider(0, 1, label="same_srv_rate", value=1),
    gr.Slider(0, 1, label="diff_srv_rate", value=0),
    gr.Slider(0, 1, label="srv_diff_host_rate", value=0),
    gr.Number(label="dst_host_count", value=0),
    gr.Number(label="dst_host_srv_count", value=0),
    gr.Slider(0, 1, label="dst_host_same_srv_rate", value=1),
    gr.Slider(0, 1, label="dst_host_diff_srv_rate", value=0),
    gr.Slider(0, 1, label="dst_host_same_src_port_rate", value=0),
    gr.Slider(0, 1, label="dst_host_srv_diff_host_rate", value=0),
    gr.Slider(0, 1, label="dst_host_serror_rate", value=0),
    gr.Slider(0, 1, label="dst_host_srv_serror_rate", value=0),
    gr.Slider(0, 1, label="dst_host_rerror_rate", value=0),
    gr.Slider(0, 1, label="dst_host_srv_rerror_rate", value=0)
]

outputs = gr.Textbox(label="Classification Result")

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="🤖 Network Intrusion Detection System",
    description="Enter the network connection features to predict whether the traffic is 'normal' or an 'anomaly'. This tool uses a PyTorch-based neural network to classify network behavior.",
    allow_flagging="never"
)

demo.launch()