import torch

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch):
    batch = [i[0] for i in batch]
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out1, out2 = None, None
        batch1 = [b for i, b in enumerate(batch) if i % 2 == 0]
        batch2 = [b for i, b in enumerate(batch) if i % 2 == 1]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel1 = sum([x.numel() for x in batch1])
            storage1 = elem.storage()._new_shared(numel1)
            out1 = elem.new(storage1)

            numel2 = sum([x.numel() for x in batch2])
            storage2 = elem.storage()._new_shared(numel2)
            out2 = elem.new(storage2)
        return torch.stack(batch1, 0, out=out1), torch.stack(batch2, 0, out=out2)

    raise TypeError(default_collate_err_msg_format.format(elem_type))
