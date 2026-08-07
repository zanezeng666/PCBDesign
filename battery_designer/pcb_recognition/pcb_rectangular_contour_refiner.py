"""PCB 轮廓正交化精修器（边界剖面分析法）

算法：
  1. 对每条边扫描边界剖面，找到外接矩形
  2. 检测凹槽（边界偏离外缘的连续区段）
  3. 对凹槽凹角检测是否圆角
  4. 构建 正交多边形（矩形+矩形凹槽+可选圆角）
"""

from __future__ import annotations

import cv2
import numpy as np

from ..logger import get_logger

_log = get_logger(__name__)


class PCBRectangularContourRefiner:
    """PCB mask → 正交多边形（矩形+矩形凹槽+可选圆角凹角）"""

    def __init__(
        self,
        groove_depth_ratio: float = 0.01,
        fillet_min_radius_px: int = 3,
        fillet_max_radius_px: int = 30,
        **kwargs,
    ):
        self.groove_depth_ratio = groove_depth_ratio
        self.fillet_min_radius_px = fillet_min_radius_px
        self.fillet_max_radius_px = fillet_max_radius_px

    def refine(self, pcb_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _log.info("=" * 60)
        _log.info("PCB轮廓正交化精修 (边界剖面法): 开始")
        _log.info("=" * 60)

        h, w = pcb_mask.shape[:2]

        # 清理 mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(pcb_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 计算边界剖面
        profile = self._compute_boundary_profile(mask)
        if profile is None:
            _log.warning("无法计算边界剖面，返回原始mask")
            return pcb_mask, np.array([], dtype=np.int32)

        bbox = profile["bbox"]
        _log.info(
            "PCB外接矩形: (%d,%d)-(%d,%d) = %dx%d",
            bbox[0], bbox[1], bbox[2], bbox[3],
            bbox[2] - bbox[0], bbox[3] - bbox[1],
        )

        # 检测凹槽
        min_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        groove_threshold = max(int(min_side * self.groove_depth_ratio), 5)
        grooves = self._detect_grooves(profile, groove_threshold)
        _log.info("检测到 %d 个凹槽", len(grooves))

        # 检测圆角凹角
        for g in grooves:
            g["fillets"] = self._detect_fillet_corners(mask, g)

        # 对称化：将对称位置的凹槽参数取平均（PCB设计追求对称美观）
        self._symmetrize_grooves(grooves, bbox, profile)

        # 构建正交多边形
        contour = self._build_polygon(bbox, grooves)

        # 生成 mask
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        if len(contour) >= 3:
            cv2.fillPoly(refined_mask, [contour], 255)

        raw_area = int(np.sum(mask > 0))
        refined_area = int(np.sum(refined_mask > 0))
        _log.info(
            "面积: %d → %d (%.1f%%)",
            raw_area, refined_area,
            100.0 * refined_area / max(raw_area, 1) - 100,
        )

        _log.info("=" * 60)
        _log.info("PCB轮廓正交化精修: 完成")
        _log.info("=" * 60)

        return refined_mask, contour.reshape(-1, 1, 2).astype(np.int32)

    # ──────────────────────────────────────────────
    #  边界剖面
    # ──────────────────────────────────────────────

    def _compute_boundary_profile(self, mask: np.ndarray) -> dict | None:
        """计算四条边的边界剖面

        bbox 使用百分位数（p2/p98）而非 raw min/max，
        避免少量噪声像素（如 y=0 或 y=H-1 的杂散绿色像素）
        将外接矩形撑大数倍，导致凹槽物理过滤阈值虚高。
        """
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        ys = np.where(rows)[0]
        xs = np.where(cols)[0]
        if len(ys) == 0 or len(xs) == 0:
            return None

        # raw 范围用于剖面扫描（不遗漏数据）
        raw_min_x, raw_max_x = int(xs[0]), int(xs[-1])
        raw_min_y, raw_max_y = int(ys[0]), int(ys[-1])

        # 用向量化方式计算 top/bottom
        mask_bin = mask > 0
        top = np.full(mask.shape[1], -1, dtype=np.int32)
        bottom = np.full(mask.shape[1], -1, dtype=np.int32)
        for x in range(raw_min_x, raw_max_x + 1):
            col = np.where(mask_bin[:, x])[0]
            if len(col) > 0:
                top[x] = col[0]
                bottom[x] = col[-1]

        left = np.full(mask.shape[0], -1, dtype=np.int32)
        right = np.full(mask.shape[0], -1, dtype=np.int32)
        for y in range(raw_min_y, raw_max_y + 1):
            row = np.where(mask_bin[y, :])[0]
            if len(row) > 0:
                left[y] = row[0]
                right[y] = row[-1]

        # 鲁棒 bbox：用百分位数忽略噪声像素
        # top/left: 噪声使值偏小 → 用 p5
        # bottom/right: 噪声使值偏大 → 用 p95
        top_valid = top[top >= 0]
        bottom_valid = bottom[bottom >= 0]
        left_valid = left[left >= 0]
        right_valid = right[right >= 0]
        if len(top_valid) >= 20 and len(left_valid) >= 20:
            min_x = int(np.percentile(left_valid, 5))
            max_x = int(np.percentile(right_valid, 95))
            min_y = int(np.percentile(top_valid, 5))
            max_y = int(np.percentile(bottom_valid, 95))
        else:
            min_x, min_y = raw_min_x, raw_min_y
            max_x, max_y = raw_max_x, raw_max_y

        return {
            "bbox": (min_x, min_y, max_x, max_y),
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }

    # ──────────────────────────────────────────────
    #  凹槽检测
    # ──────────────────────────────────────────────

    def _detect_grooves(self, profile: dict, threshold: int) -> list[dict]:
        """检测长边上的凹槽

        只在PCB的两条长边上检测凹槽。
        用宽度和平坦度过滤噪声：
          - 宽度 < 30px → 噪声
          - 平坦度 > 0.5（底部不平坦）→ 角落过渡，非真实凹槽
        """
        bbox = profile["bbox"]
        min_x, min_y, max_x, max_y = bbox
        w_px = max_x - min_x
        h_px = max_y - min_y

        # 只在长边上检测凹槽
        if w_px >= h_px:
            long_edges = ["top", "bottom"]
        else:
            long_edges = ["left", "right"]

        min_width = 30
        max_flatness = 0.5  # 平坦度阈值
        max_depth_ratio = 0.35  # 凹槽深度不超过短边的35%

        grooves: list[dict] = []
        for edge in long_edges:
            if edge == "top":
                raw = self._scan_for_grooves(
                    profile["top"], min_x, max_x, "top", "min", threshold, bbox,
                )
            elif edge == "bottom":
                raw = self._scan_for_grooves(
                    profile["bottom"], min_x, max_x, "bottom", "max", threshold, bbox,
                )
            elif edge == "left":
                raw = self._scan_for_grooves(
                    profile["left"], min_y, max_y, "left", "min", threshold, bbox,
                )
            else:
                raw = self._scan_for_grooves(
                    profile["right"], min_y, max_y, "right", "max", threshold, bbox,
                )

            for g in raw:
                if g["width"] < min_width:
                    continue
                if g.get("flatness", 0) > max_flatness:
                    _log.info("  过滤[%s]: 平坦度=%.2f (角落过渡)", edge, g["flatness"])
                    continue
                if g["depth"] > max_depth_ratio * min(w_px, h_px):
                    _log.info("  过滤[%s]: 深度=%d > %dpx (角落过渡)", edge, g["depth"], int(max_depth_ratio * min(w_px, h_px)))
                    continue

                # 物理尺寸过滤：太小的凹槽不可能是真实PCB设计特征
                if edge in ("top", "bottom"):
                    min_depth_px = h_px * 0.03
                    min_width_px = w_px * 0.10
                else:
                    min_depth_px = w_px * 0.02
                    min_width_px = h_px * 0.10
                if g["depth"] < min_depth_px or g["width"] < min_width_px:
                    _log.info(
                        "  过滤[%s]: 物理尺寸过小 深度=%dpx(<%.1f) 宽度=%dpx(<%.1f) — 肉眼不可见",
                        edge, g["depth"], min_depth_px, g["width"], min_width_px,
                    )
                    continue

                grooves.append(g)

        return grooves

    def _scan_for_grooves(
        self, boundary: np.ndarray, lo: int, hi: int,
        edge: str, direction: str, threshold: int, bbox: tuple,
    ) -> list[dict]:
        """在单条边上扫描凹槽

        用分位数法确定外缘位置，然后检测偏离外缘的连续区段。
        对宽区段在局部最小值处拆分为独立凹槽。
        用凹槽底部平坦度区分真实凹槽和角落过渡。
        """
        results = []
        vals, xs = [], []
        for i in range(lo, hi + 1):
            if boundary[i] >= 0:
                vals.append(boundary[i])
                xs.append(i)
        if not vals:
            return []

        vals_arr = np.array(vals, dtype=np.float64)
        xs_arr = np.array(xs, dtype=np.float64)

        # 外缘位置（用分位数抗噪）
        if direction == "min":
            outer = float(np.percentile(vals_arr, 5))
            deviations = vals_arr - outer
        else:
            outer = float(np.percentile(vals_arr, 95))
            deviations = outer - vals_arr

        # 找连续偏离区段
        in_groove = deviations > threshold
        i, n = 0, len(in_groove)
        while i < n:
            if in_groove[i]:
                j = i
                while j < n and in_groove[j]:
                    j += 1

                # 尝试在局部最小值处拆分宽区段
                seg_devs = deviations[i:j]
                seg_xs = xs_arr[i:j]
                seg_vals = vals_arr[i:j]
                sub_segs = self._split_at_local_minima(
                    seg_devs, seg_xs, seg_vals, threshold, direction,
                )

                for sub_devs, sub_xs, sub_vals in sub_segs:
                    depth = float(np.max(sub_devs))
                    if direction == "min":
                        groove_pos = int(np.max(sub_vals))
                    else:
                        groove_pos = int(np.min(sub_vals))

                    if depth > 0:
                        flatness = np.std(sub_vals) / depth
                    else:
                        flatness = 1.0

                    results.append({
                        "edge": edge,
                        "start": int(sub_xs[0]),
                        "end": int(sub_xs[-1]),
                        "outer_pos": int(outer),
                        "groove_pos": groove_pos,
                        "depth": int(depth),
                        "width": int(sub_xs[-1] - sub_xs[0]),
                        "flatness": float(flatness),
                    })
                    _log.info(
                        "  凹槽候选[%s]: [%d,%d] 深度=%d 宽度=%d 平坦度=%.2f",
                        edge, int(sub_xs[0]), int(sub_xs[-1]),
                        int(depth), int(sub_xs[-1] - sub_xs[0]), float(flatness),
                    )
                i = j
            else:
                i += 1
        return results

    @staticmethod
    def _split_at_local_minima(
        devs: np.ndarray, xs: np.ndarray, vals: np.ndarray,
        threshold: int, direction: str,
    ) -> list[tuple]:
        """在局部最小值处拆分宽凹槽段为独立子段。

        当一个连续偏离区段内部存在偏离值显著下降的位置时，
        说明可能是多个相邻凹槽被合并，在此处拆分。
        拆分条件：偏离值降至 max_dev 的 50% 以下，且持续 >= 3px。
        """
        if len(devs) < 10:
            return [(devs, xs, vals)]

        max_dev = float(np.max(devs))
        if max_dev <= 0:
            return [(devs, xs, vals)]

        # 拆分阈值：偏离值低于 max_dev 的 50%
        split_level = max_dev * 0.5
        if split_level <= threshold:
            return [(devs, xs, vals)]

        below = devs < split_level
        if not np.any(below):
            return [(devs, xs, vals)]

        # 找连续的 below 区段（潜在分割点）
        sub_segs = []
        seg_start = 0
        k = 0
        while k < len(devs):
            if below[k]:
                # 找连续 below 区段
                m = k
                while m < len(devs) and below[m]:
                    m += 1
                gap_len = m - k
                if gap_len >= 3:
                    # 在 gap 中点拆分
                    split_pt = k + gap_len // 2
                    if split_pt - seg_start >= 5:  # 前段至少 5px
                        sub_segs.append((devs[seg_start:split_pt],
                                         xs[seg_start:split_pt],
                                         vals[seg_start:split_pt]))
                    seg_start = m  # 后段从 gap 结束开始
                k = m
            else:
                k += 1

        # 添加最后一段
        if seg_start < len(devs) and len(devs) - seg_start >= 5:
            sub_segs.append((devs[seg_start:], xs[seg_start:], vals[seg_start:]))

        return sub_segs if sub_segs else [(devs, xs, vals)]

    # ──────────────────────────────────────────────
    #  对称化
    # ──────────────────────────────────────────────

    def _symmetrize_grooves(self, grooves: list[dict], bbox: tuple, profile: dict) -> None:
        """对称化凹槽：PCB设计通常追求对称美观。

        首先检测PCB整体是否上下/左右对称，若对称则：
          1. 放宽凹槽匹配容差（3倍），使更多凹槽配对
          2. 配对的凹槽参数取平均，拟合成完全相同
        注意：不会自动为未配对的凹槽创建镜像——如果判定对称，
        必然能在对应位置找到匹配的凹槽（仅尺寸有细微差异）。
        """
        # 检测整体对称性
        symmetry = self._check_overall_symmetry(profile, bbox)

        min_x, min_y, max_x, max_y = bbox
        w_px = max_x - min_x
        h_px = max_y - min_y
        base_tol = max(int(0.02 * max(w_px, h_px)), 8)

        by_edge: dict[str, list[int]] = {"top": [], "bottom": [], "left": [], "right": []}
        for i, g in enumerate(grooves):
            by_edge[g["edge"]].append(i)

        used: set[int] = set()

        # 1. top ↔ bottom
        h_tol = base_tol * 3 if symmetry["h_symmetric"] else base_tol
        for i in by_edge["top"]:
            if i in used:
                continue
            for j in by_edge["bottom"]:
                if j in used:
                    continue
                gi, gj = grooves[i], grooves[j]
                if (abs(gi["start"] - gj["start"]) < h_tol and
                        abs(gi["end"] - gj["end"]) < h_tol):
                    self._sync_pair(gi, gj, sync_position=True)
                    used.update({i, j})
                    _log.info(
                        "  对称化: top[%d,%d] <-> bottom[%d,%d] -> start=%d end=%d depth=%d",
                        gi["start"], gi["end"], gj["start"], gj["end"],
                        gi["start"], gi["end"], gi["depth"],
                    )
                    break

        # 2. left ↔ right
        v_tol = base_tol * 3 if symmetry["v_symmetric"] else base_tol
        for i in by_edge["left"]:
            if i in used:
                continue
            for j in by_edge["right"]:
                if j in used:
                    continue
                gi, gj = grooves[i], grooves[j]
                if (abs(gi["start"] - gj["start"]) < v_tol and
                        abs(gi["end"] - gj["end"]) < v_tol):
                    self._sync_pair(gi, gj, sync_position=True)
                    used.update({i, j})
                    _log.info(
                        "  对称化: left[%d,%d] <-> right[%d,%d] -> start=%d end=%d depth=%d",
                        gi["start"], gi["end"], gj["start"], gj["end"],
                        gi["start"], gi["end"], gi["depth"],
                    )
                    break

        # 3. 同边镜像
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for edge_name, indices in by_edge.items():
            center = cx if edge_name in ("top", "bottom") else cy
            for ii in range(len(indices)):
                i = indices[ii]
                if i in used:
                    continue
                for jj in range(ii + 1, len(indices)):
                    j = indices[jj]
                    if j in used:
                        continue
                    gi, gj = grooves[i], grooves[j]
                    if (abs(gi["start"] - (2 * center - gj["end"])) < base_tol and
                            abs(gi["end"] - (2 * center - gj["start"])) < base_tol):
                        self._sync_pair(gi, gj, sync_position=False)
                        used.update({i, j})
                        _log.info(
                            "  对称化: %s镜像 [%d,%d] <-> [%d,%d] (depth=%d)",
                            edge_name, gi["start"], gi["end"],
                            gj["start"], gj["end"], gi["depth"],
                        )
                        break

    def _check_overall_symmetry(self, profile: dict, bbox: tuple) -> dict:
        """检测PCB整体是否上下/左右对称。

        比较top与bottom（或left与right）的边界剖面，如果镜像后平均偏差
        小于阈值的5%（至少8px），判定为对称。
        """
        min_x, min_y, max_x, max_y = bbox
        w_px = max_x - min_x
        h_px = max_y - min_y
        result = {"h_symmetric": False, "v_symmetric": False}

        # 上下对称：top偏离min_y vs bottom偏离max_y
        top_arr = profile["top"][min_x:max_x + 1].astype(np.float64)
        bottom_arr = profile["bottom"][min_x:max_x + 1].astype(np.float64)
        top_off = top_arr - min_y
        bottom_off = max_y - bottom_arr
        valid = (top_arr >= 0) & (bottom_arr >= 0)
        if np.sum(valid) > w_px * 0.5:
            diff = np.abs(top_off[valid] - bottom_off[valid])
            mean_diff = float(np.mean(diff))
            threshold = max(h_px * 0.05, 8)
            if mean_diff < threshold:
                result["h_symmetric"] = True
                _log.info("  PCB整体上下对称 (平均偏差=%.1fpx < %.1fpx)", mean_diff, threshold)

        # 左右对称：left偏离min_x vs right偏离max_x
        left_arr = profile["left"][min_y:max_y + 1].astype(np.float64)
        right_arr = profile["right"][min_y:max_y + 1].astype(np.float64)
        left_off = left_arr - min_x
        right_off = max_x - right_arr
        valid = (left_arr >= 0) & (right_arr >= 0)
        if np.sum(valid) > h_px * 0.5:
            diff = np.abs(left_off[valid] - right_off[valid])
            mean_diff = float(np.mean(diff))
            threshold = max(w_px * 0.05, 8)
            if mean_diff < threshold:
                result["v_symmetric"] = True
                _log.info("  PCB整体左右对称 (平均偏差=%.1fpx < %.1fpx)", mean_diff, threshold)

        return result

    def _sync_pair(self, g1: dict, g2: dict, sync_position: bool) -> None:
        """将两个凹槽参数同步（取平均）

        sync_position=True: 同步位置(同边/对面) — start/end/width/depth/fillets 全部平均
        sync_position=False: 仅同步深度和圆角(同边镜像) — 保留各自位置
        """
        avg_depth = (g1["depth"] + g2["depth"]) // 2

        if sync_position:
            avg_start = (g1["start"] + g2["start"]) // 2
            avg_end = (g1["end"] + g2["end"]) // 2
            for g in (g1, g2):
                g["start"] = avg_start
                g["end"] = avg_end
                g["width"] = avg_end - avg_start

        for g in (g1, g2):
            g["depth"] = avg_depth
            if g["edge"] in ("top", "left"):
                g["groove_pos"] = g["outer_pos"] + avg_depth
            else:
                g["groove_pos"] = g["outer_pos"] - avg_depth

        # 同步圆角
        f1 = {f["position"]: f for f in g1.get("fillets", [])}
        f2 = {f["position"]: f for f in g2.get("fillets", [])}
        avg_fillets = []
        for pos in ("start", "end"):
            r1 = f1.get(pos, {}).get("radius")
            r2 = f2.get(pos, {}).get("radius")
            if r1 is not None and r2 is not None:
                avg_fillets.append({"position": pos, "radius": (r1 + r2) // 2})
            elif r1 is not None:
                avg_fillets.append({"position": pos, "radius": r1})
            elif r2 is not None:
                avg_fillets.append({"position": pos, "radius": r2})
        g1["fillets"] = list(avg_fillets)
        g2["fillets"] = list(avg_fillets)

    # ──────────────────────────────────────────────
    #  圆角凹角检测
    # ──────────────────────────────────────────────

    def _detect_fillet_corners(self, mask: np.ndarray, groove: dict) -> list[dict]:
        """检测凹槽两端凹角是否圆角（过渡宽度法）

        过渡宽度 ≈ 圆角半径。
        圆角半径不超过凹槽深度的80%（物理约束）。
        """
        fillets = []
        edge = groove["edge"]
        outer = groove["outer_pos"]
        groove_y = groove["groove_pos"]
        depth = abs(groove_y - outer)
        if depth < 3:
            return fillets

        # 圆角最大半径 = 凹槽深度的80%
        max_r = min(int(depth * 0.8), self.fillet_max_radius_px)

        if edge in ("top", "bottom"):
            for pos_name, corner_x in [
                ("start", groove["start"]), ("end", groove["end"]),
            ]:
                r = self._measure_transition_h(
                    mask, corner_x, edge, outer, groove_y, depth,
                )
                r = min(r, max_r)  # 物理约束
                if r >= self.fillet_min_radius_px:
                    fillets.append({"position": pos_name, "radius": r})
                    _log.info("    %s角: 圆角 r=%dpx", pos_name, r)
                else:
                    _log.info("    %s角: 直角 (过渡=%dpx)", pos_name, r)
        else:
            for pos_name, corner_y in [
                ("start", groove["start"]), ("end", groove["end"]),
            ]:
                r = self._measure_transition_v(
                    mask, corner_y, edge, outer, groove_y, depth,
                )
                r = min(r, max_r)
                if r >= self.fillet_min_radius_px:
                    fillets.append({"position": pos_name, "radius": r})
                    _log.info("    %s角: 圆角 r=%dpx", pos_name, r)
                else:
                    _log.info("    %s角: 直角 (过渡=%dpx)", pos_name, r)

        return fillets

    def _measure_transition_h(
        self, mask: np.ndarray, corner_x: int,
        edge: str, outer: int, groove_y: int, depth: int,
    ) -> int:
        """测量水平边凹角处的过渡宽度"""
        h, w = mask.shape
        mask_bin = mask > 0
        search_r = self.fillet_max_radius_px

        ys = []
        for dx in range(-search_r, search_r + 1):
            x = corner_x + dx
            if x < 0 or x >= w:
                continue
            col = np.where(mask_bin[:, x])[0]
            if len(col) == 0:
                continue
            by = int(col[0]) if edge == "top" else int(col[-1])
            ys.append(by)

        if len(ys) < 5:
            return 0

        ys_arr = np.array(ys)
        # 偏离比例: 0=外缘, 1=凹槽底
        if edge == "top":
            ratios = (ys_arr - outer) / max(depth, 1)
        else:
            ratios = (outer - ys_arr) / max(depth, 1)

        # 过渡点: 0.1 < ratio < 0.9
        in_transition = (ratios > 0.1) & (ratios < 0.9)
        count = int(np.sum(in_transition))
        return count

    def _measure_transition_v(
        self, mask: np.ndarray, corner_y: int,
        edge: str, outer: int, groove_y: int, depth: int,
    ) -> int:
        """测量垂直边凹角处的过渡宽度"""
        h, w = mask.shape
        mask_bin = mask > 0
        search_r = self.fillet_max_radius_px

        xs = []
        for dy in range(-search_r, search_r + 1):
            y = corner_y + dy
            if y < 0 or y >= h:
                continue
            row = np.where(mask_bin[y, :])[0]
            if len(row) == 0:
                continue
            bx = int(row[0]) if edge == "left" else int(row[-1])
            xs.append(bx)

        if len(xs) < 5:
            return 0

        xs_arr = np.array(xs)
        if edge == "left":
            ratios = (xs_arr - outer) / max(depth, 1)
        else:
            ratios = (outer - xs_arr) / max(depth, 1)

        in_transition = (ratios > 0.1) & (ratios < 0.9)
        count = int(np.sum(in_transition))
        return count

    # ──────────────────────────────────────────────
    #  构建正交多边形
    # ──────────────────────────────────────────────

    def _build_polygon(
        self, bbox: tuple, grooves: list[dict],
    ) -> np.ndarray:
        """构建正交多边形：外接矩形 - 矩形凹槽（含可选圆角）"""
        min_x, min_y, max_x, max_y = bbox

        # 按边分组
        by_edge = {"top": [], "bottom": [], "left": [], "right": []}
        for g in grooves:
            by_edge[g["edge"]].append(g)

        pts: list[tuple[int, int]] = [(min_x, min_y)]

        # 上边（左→右）
        self._add_horizontal(pts, min_x, max_x, min_y, by_edge["top"], "top", 1)
        pts.append((max_x, min_y))
        # 右边（上→下）
        self._add_vertical(pts, min_y, max_y, max_x, by_edge["right"], "right", 1)
        pts.append((max_x, max_y))
        # 下边（右→左）
        self._add_horizontal(pts, max_x, min_x, max_y, by_edge["bottom"], "bottom", -1)
        pts.append((min_x, max_y))
        # 左边（下→上）
        self._add_vertical(pts, max_y, min_y, min_x, by_edge["left"], "left", -1)

        # 闭合 + 去重
        pts.append((min_x, min_y))
        result = [pts[0]]
        for p in pts[1:]:
            if p != result[-1]:
                result.append(p)
        if len(result) > 2 and result[0] == result[-1]:
            result.pop()

        return np.array(result, dtype=np.int32)

    def _add_horizontal(
        self, pts: list, start_x: int, end_x: int, y_outer: int,
        grooves: list[dict], edge: str, direction: int,
    ):
        """在水平边上插入凹槽"""
        grooves = sorted(grooves, key=lambda g: g["start"])
        if direction == -1:
            grooves = list(reversed(grooves))

        for g in grooves:
            if direction == 1:
                self._add_groove_h(pts, g, y_outer, edge, is_lr=True)
            else:
                self._add_groove_h(pts, g, y_outer, edge, is_lr=False)

    def _add_groove_h(
        self, pts: list, g: dict, y_outer: int, edge: str, is_lr: bool,
    ):
        """添加单个水平边凹槽（含圆角处理）

        路径结构（有圆角时）：
          外缘 → 壁切点 → [弧] → 底切点 → ... → 底切点 → [弧] → 壁切点 → 外缘
        无圆角时直接用尖角点。
        """
        g_start = g["start"]
        g_end = g["end"]
        g_y = g["groove_pos"]
        fillets = {f.get("position"): f for f in g.get("fillets", [])}

        if edge == "top":
            # top边: groove_y > outer_y (凹槽向下)
            sign = 1  # groove_y - r 在 outer 和 groove_y 之间
        else:
            # bottom边: groove_y < outer_y (凹槽向上)
            sign = -1

        if is_lr:
            # 从左到右
            pts.append((g_start, y_outer))

            # 入口角
            f = fillets.get("start")
            if f:
                r = f["radius"]
                pts.append((g_start, g_y - sign * r))
                cx, cy = g_start + r, g_y - sign * r
                a0, a1 = (np.pi, np.pi / 2) if sign > 0 else (0, -np.pi / 2)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_start + r, g_y))
            else:
                pts.append((g_start, g_y))

            # 出口角
            f = fillets.get("end")
            if f:
                r = f["radius"]
                pts.append((g_end - r, g_y))
                cx, cy = g_end - r, g_y - sign * r
                a0, a1 = (np.pi / 2, 0) if sign > 0 else (-np.pi / 2, -np.pi)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_end, g_y - sign * r))
            else:
                pts.append((g_end, g_y))

            pts.append((g_end, y_outer))
        else:
            # 从右到左
            pts.append((g_end, y_outer))

            # 入口角（先碰到 g_end）
            f = fillets.get("end")
            if f:
                r = f["radius"]
                pts.append((g_end, g_y - sign * r))
                cx, cy = g_end - r, g_y - sign * r
                # sign>0(top): wall tangent at angle π; sign<0(bottom): at angle 0
                a0, a1 = (np.pi, np.pi / 2) if sign > 0 else (0, -np.pi / 2)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_end - r, g_y))
            else:
                pts.append((g_end, g_y))

            # 出口角（后碰到 g_start）
            f = fillets.get("start")
            if f:
                r = f["radius"]
                pts.append((g_start + r, g_y))
                cx, cy = g_start + r, g_y - sign * r
                # sign>0(top): bottom tangent at angle π/2; sign<0(bottom): at angle -π/2
                a0, a1 = (np.pi / 2, 0) if sign > 0 else (-np.pi / 2, -np.pi)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_start, g_y - sign * r))
            else:
                pts.append((g_start, g_y))

            pts.append((g_start, y_outer))

    def _add_vertical(
        self, pts: list, start_y: int, end_y: int, x_outer: int,
        grooves: list[dict], edge: str, direction: int,
    ):
        """在垂直边上插入凹槽"""
        grooves = sorted(grooves, key=lambda g: g["start"])
        if direction == -1:
            grooves = list(reversed(grooves))

        for g in grooves:
            if direction == 1:
                self._add_groove_v(pts, g, x_outer, edge, is_tb=True)
            else:
                self._add_groove_v(pts, g, x_outer, edge, is_tb=False)

    def _add_groove_v(
        self, pts: list, g: dict, x_outer: int, edge: str, is_tb: bool,
    ):
        """添加单个垂直边凹槽（含圆角处理）"""
        g_start = g["start"]
        g_end = g["end"]
        g_x = g["groove_pos"]
        fillets = {f.get("position"): f for f in g.get("fillets", [])}

        if edge == "left":
            sign = 1  # groove_x > outer_x (凹槽向右)
        else:
            sign = -1  # groove_x < outer_x (凹槽向左)

        if is_tb:
            # 从上到下
            pts.append((x_outer, g_start))

            f = fillets.get("start")
            if f:
                r = f["radius"]
                pts.append((g_x - sign * r, g_start))
                cx, cy = g_x - sign * r, g_start + r
                a0, a1 = (0, np.pi / 2) if sign > 0 else (np.pi, np.pi / 2)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_x, g_start + r))
            else:
                pts.append((g_x, g_start))

            f = fillets.get("end")
            if f:
                r = f["radius"]
                pts.append((g_x, g_end - r))
                cx, cy = g_x - sign * r, g_end - r
                a0, a1 = (np.pi / 2, np.pi) if sign > 0 else (np.pi / 2, 0)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_x - sign * r, g_end))
            else:
                pts.append((g_x, g_end))

            pts.append((x_outer, g_end))
        else:
            # 从下到上
            pts.append((x_outer, g_end))

            f = fillets.get("end")
            if f:
                r = f["radius"]
                pts.append((g_x - sign * r, g_end))
                cx, cy = g_x - sign * r, g_end - r
                a0, a1 = (np.pi, np.pi / 2) if sign > 0 else (0, np.pi / 2)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_x, g_end - r))
            else:
                pts.append((g_x, g_end))

            f = fillets.get("start")
            if f:
                r = f["radius"]
                pts.append((g_x, g_start + r))
                cx, cy = g_x - sign * r, g_start + r
                a0, a1 = (np.pi / 2, 0) if sign > 0 else (np.pi / 2, np.pi)
                for a in np.linspace(a0, a1, max(5, r // 2)):
                    pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
                pts.append((g_x - sign * r, g_start))
            else:
                pts.append((g_x, g_start))

            pts.append((x_outer, g_start))


if __name__ == "__main__":
    refiner = PCBRectangularContourRefiner()
    print("PCB轮廓正交化精修器 (边界剖面法) 初始化完成")
