from collections import defaultdict
from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

from skidl.pin import pin_types

SKIDL_lib_version = '0.0.1'

schematic = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'DW01-G', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'DW01-G'}), 'ref_prefix':'U', 'fplist':['Package_TO_SOT_SMD:SOT-23-6'], 'footprint':'Package_TO_SOT_SMD:SOT-23-6', 'keywords':'', 'description':'', 'datasheet':'', 'pins':[
            Pin(num='1',name='OD',func=pin_types.PWRIN),
            Pin(num='2',name='CS',func=pin_types.PASSIVE),
            Pin(num='3',name='OC',func=pin_types.PWROUT),
            Pin(num='4',name='TD',func=pin_types.INPUT),
            Pin(num='5',name='VDD',func=pin_types.PWRIN),
            Pin(num='6',name='VSS',func=pin_types.PWRIN)], 'unit_defs':[] }),
        Part(**{ 'name':'FS8205A', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'FS8205A'}), 'ref_prefix':'Q', 'fplist':['Package_SO:TSSOP-8_4.4x3mm_P0.65mm'], 'footprint':'Package_SO:TSSOP-8_4.4x3mm_P0.65mm', 'keywords':'', 'description':'', 'datasheet':'', 'pins':[
            Pin(num='1',name='S1',func=pin_types.PASSIVE),
            Pin(num='2',name='G1',func=pin_types.INPUT),
            Pin(num='3',name='S2',func=pin_types.PASSIVE),
            Pin(num='4',name='G2',func=pin_types.INPUT),
            Pin(num='5',name='D2',func=pin_types.PASSIVE),
            Pin(num='6',name='D2',func=pin_types.PASSIVE),
            Pin(num='7',name='D1',func=pin_types.PASSIVE),
            Pin(num='8',name='D1',func=pin_types.PASSIVE)], 'unit_defs':[] }),
        Part(**{ 'name':'R_Small_US', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'R_Small_US'}), 'ref_prefix':'R', 'fplist':[''], 'footprint':'Resistor_SMD:R_0603_1608Metric', 'keywords':'r resistor', 'description':'Resistor, small US symbol', 'datasheet':'~', 'pins':[
            Pin(num='1',name='~',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='~',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'C_Small', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'C_Small'}), 'ref_prefix':'C', 'fplist':[''], 'footprint':'Capacitor_SMD:C_0603_1608Metric', 'keywords':'capacitor cap', 'description':'Unpolarized capacitor, small symbol', 'datasheet':'~', 'pins':[
            Pin(num='1',name='~',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='~',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'Conn_01x04', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Conn_01x04'}), 'ref_prefix':'J', 'fplist':[''], 'footprint':'Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical', 'keywords':'connector', 'description':'Generic connector, single row, 01x04, script generated (kicad-library-utils/schlib/autogen/connector/)', 'datasheet':'~', 'pins':[
            Pin(num='1',name='Pin_1',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='Pin_2',func=pin_types.PASSIVE,unit=1),
            Pin(num='3',name='Pin_3',func=pin_types.PASSIVE,unit=1),
            Pin(num='4',name='Pin_4',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] })])