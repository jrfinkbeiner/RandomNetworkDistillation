
import sys
import ase
from ase.db import connect
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList, RemoveOffsets, CastTo32

datapath = '../qm9.db'
qm9data = QM9(
    '../qm9.db', 
    batch_size=10,
    num_train=110000,
    num_val=10000,
    transforms=[
        ASENeighborList(cutoff=5.),
        # RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
        CastTo32()
    ],
)
qm9data.prepare_data()
qm9data.setup()

print('Number of reference calculations:', len(qm9data.dataset))
print('Number of train data:', len(qm9data.train_dataset))
print('Number of validation data:', len(qm9data.val_dataset))
print('Number of test data:', len(qm9data.test_dataset))
print('Available properties:')

for p in qm9data.dataset.available_properties:
    print('-', p)



with connect(datapath) as conn:
    print(conn.metadata)

    for key,val in conn.metadata["atomrefs"].items():
        print(key, len(val), type(val))

    # for row in conn.select(limit=4):
    #     print(row.metadata)
    # conn.metadata = {
    #     "_property_unit_dict": property_unit_dict,
    #     "_distance_unit": distance_unit,
    #     "atomrefs": atomrefs,
    # }


sys.exit()

train_dataset = qm9data.train_dataset
val_dataset = qm9data.val_dataset
test_dataset = qm9data.test_dataset

print("len train:", len(train_dataset))
print("len val:", len(val_dataset))
print("len test:", len(test_dataset))
print(train_dataset[0])
print(type(train_dataset))


print("\n--------------- Predictions ------------------")
rcut = 5.0 # LiCl: 5.5, NaCl: 6.0, KCl: 7.0
model_type = "large-ext"

from inp_helper import get_target_model
model = get_target_model(model_type, rcut)

for batch in qm9data.test_dataloader():
    print(batch)
    # break
    result = model(batch)
    print("Result dictionary:", result)
    print("\n----------------------------------------------------")
    for key,item in result.items():
        print()
        print(key,item)
    break