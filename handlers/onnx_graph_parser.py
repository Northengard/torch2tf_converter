import onnx
from onnx import numpy_helper
from onnxsim import simplify


class OnnxInterModel:
    def __init__(self, onnx_path, use_simplify=False):
        self._onnx_model = onnx.load(onnx_path)
        if use_simplify:
            self._onnx_model, check = simplify(self._onnx_model)
            assert check, "Simplified ONNX model could not be validated"

        self._model_graph = self._onnx_model.graph
        self._model_weights = {w.name: numpy_helper.to_array(w) for w in self._model_graph.initializer}

        self._model_input = dict([self._parse_input(inp) for inp in self._model_graph.input])
        self._model_output = dict([self._parse_input(out) for out in self._model_graph.output])
        self.nodes, self.adj_list = self.parse_model()

    @property
    def model_input(self):
        return self._model_input

    @property
    def model_output(self):
        return self._model_output

    def parse_node(self, node):
        node_attrs = getattr(self, f'_get_{node.op_type.lower()}')(node)
        node_attrs['name'] = node.name
        node_attrs['op_type'] = node.op_type.lower()
        return node_attrs

    def parse_model(self):
        inputs = dict()
        nodes = list()
        adj_list = dict()
        helper_node = None
        for node_id, node in enumerate(self._model_graph.node):
            node_attrs = self.parse_node(node)
            node_attrs['node_id'] = node_id
            if (helper_node is not None) and (node_attrs['op_type'] == 'upsample'):
                node_attrs['input'].remove(helper_node['output'][0])
                node_attrs['scale'] = helper_node['values'][2:]
                helper_node = None
            if len(node_attrs['input']) < 1:
                helper_node = node_attrs
                continue
            for inp in node_attrs['input']:
                if inp in inputs.keys():
                    inputs[inp].append(node_id)
                else:
                    inputs[inp] = [node_id]
            nodes.append(node_attrs)

        for node in nodes:
            adj_list[node['node_id']] = [inp for link in node['output'] if link in inputs.keys()
                                         for inp in inputs[link]]
        return nodes, adj_list

    @staticmethod
    def _parse_input(input_obj):
        return input_obj.name, [dim.dim_value for dim in input_obj.type.tensor_type.shape.dim]

    @staticmethod
    def _get_conv_attrs(node):
        attrs = {attr.name: attr.ints if attr.name != 'group' else attr.i for attr in node.attribute}
        if 'pads' in attrs.keys():
            attrs['pads'] = attrs['pads'][1::2]
        return attrs

    @staticmethod
    def _get_linear_attrs(node):
        attrs = {attr.name: attr.i if attr.type - 1 else attr.f for attr in node.attribute}
        return attrs

    def _get_weighted(self, node, attrs):
        conv_weights = [self._model_weights[w_name] for w_name in node.input[1:]]

        attrs['input'] = node.input[:1]
        attrs['weights'] = conv_weights
        attrs['output'] = node.output

    def _get_conv(self, node):
        conv_attr = self._get_conv_attrs(node)
        self._get_weighted(node, conv_attr)
        return conv_attr

    def _get_gemm(self, node):
        attrs = self._get_linear_attrs(node)
        self._get_weighted(node, attrs)
        return attrs

    @staticmethod
    def _get_node(node):
        attr_params = dict({'input': node.input, 'output': node.output})
        return attr_params

    def _get_pool(self, node):
        attrs = self._get_node(node)
        attrs.update(self._get_conv_attrs(node))
        return attrs

    def _get_maxpool(self, node):
        return self._get_pool(node)

    def _get_globalaveragepool(self, node):
        return self._get_pool(node)

    def _get_clip(self, node):
        attrs = self._get_node(node)
        attrs['values'] = sorted([attr.f for attr in node.attribute])
        return attrs

    def _get_constant(self, node):
        attrs = self._get_node(node)
        attrs['values'] = numpy_helper.to_array(node.attribute[0].t)
        return attrs

    def _get_upsample(self, node):
        attrs = self._get_node(node)
        attrs['mode'] = node.attribute[0].s.decode()
        if attrs['mode'] == 'linear':
            attrs['mode'] = 'bilinear'
        return attrs

    def _get_relu(self, node):
        return self._get_node(node)

    def _get_concat(self, node):
        return self._get_node(node)

    def _get_add(self, node):
        return self._get_node(node)

    def _get_sigmoid(self, node):
        return self._get_node(node)

    def _get_flatten(self, node):
        attrs = self._get_node(node)
        attrs['axis'] = node.attribute[0].i
        return attrs
