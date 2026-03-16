"""后台图片并发自测服务。"""

import asyncio
import json
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import Response

from ..core.config import config
from ..core.models import ChatCompletionRequest
from ..services.generation_handler import MODEL_CONFIG


PROMPT_LIBRARY: List[str] = [
    "清晨海港边停靠的一艘复古木船，薄雾与金色日出交织，电影级光影，细节丰富，禁止水印和文字",
    "雪山山脊上的未来科考站，冷蓝色环境光与暖色舱灯对比鲜明，超写实摄影风格",
    "雨夜霓虹街头的拉面小店门口，一只戴围巾的柴犬坐在凳子上，画面干净，适合做壁纸",
    "沙漠中的巨型风力发电阵列，天空云层翻涌，广角构图，强调空间纵深",
    "古代竹林书院内的少年琴师，晨光透过竹叶洒落，国风数字艺术，高细节",
    "赛博朋克都市天桥上的女侦探，远处悬浮广告屏闪烁，夜景反射，主体清晰",
    "深海研究站外缓慢游过的发光鲸鱼群，蓝绿色氛围，梦幻概念设定图",
    "初秋湖畔的玻璃咖啡馆，落叶飞舞，暖色调，纹理真实，光线自然",
    "机械城堡顶部的蒸汽钟楼，齿轮与铜管密布，蒸汽朋克插画，层次分明",
    "热带雨林中的巨大白色瀑布，阳光穿透水雾形成彩虹，画质清晰，构图完整",
    "月球基地停机坪上的运输飞船准备起飞，银灰色金属质感，精致 3D 渲染",
    "夜色中的中式古桥与灯笼倒影，小舟缓缓划过河面，柔和水彩风格",
    "高空云海上漂浮的空岛农庄，风车缓慢旋转，童话氛围，色彩统一",
    "未来医院的无菌手术舱，蓝白极简设计，近景特写，突出材质与设备细节",
    "北极冰原上的极光营地，雪地车停在前景，星空清晰，商业海报质感",
    "火山口边缘的黑曜石神殿，红色熔岩映亮天空，高对比强烈视觉冲击",
    "安静地铁车厢里的机器人乘客，车窗外闪过霓虹城市，低多边形艺术风格",
    "江南古镇的石板路与油纸伞少女，细雨朦胧，光影柔和，国风写意",
    "阳光灿烂的向日葵田中央，一辆薄荷绿色复古敞篷车停靠，色彩明快",
    "太空电梯内部的透明观景平台，俯瞰地球曲线，科幻感强，主体辨识度高",
    "森林深处长满荧光蘑菇的小路，一只白鹿回头凝望，梦幻设定图",
    "东京风格屋顶花园里的自动售货机角落，黄昏橙色天空，画面简洁高级",
    "维多利亚时代图书馆的旋转楼梯与天窗，尘埃在光束中漂浮，超写实摄影",
    "未来无人货运港口的机械臂群正在装载集装箱，秩序感强，冷色工业风",
    "荒原中的红色电话亭，四周只有风与草浪，极简构图，适合作封面",
    "夜幕下的山顶天文台，穹顶缓缓打开，银河清晰可见，细节丰富",
    "中世纪集市上的面包摊，热气腾腾，人物神态自然，暖色电影感",
    "玻璃温室中的白色孔雀展开尾羽，晨间逆光，层次丰富，纹理清晰",
    "漂浮在云层中的未来寺庙，琉璃材质与金属结构结合，神圣而科幻",
    "海底古城遗迹中的巨大石像，鱼群穿梭而过，蓝绿色环境光，沉浸感强",
    "南法小镇街角的花店门面，自行车靠在墙边，午后阳光温暖，写实摄影",
    "高原草甸上的白色帐篷营地，远处雪山倒映在湖面，商业宣传图质感",
    "复古计算机实验室内堆满显像管设备，绿色终端光照亮空间，怀旧科技风",
    "穿越峡谷的悬索列车，晨雾环绕桥体，广角构图，强调速度与尺度",
    "中国神话风格的凤凰停驻在青铜神树上，火焰与霞光交织，国风史诗感",
    "暴雨前的海边公路，乌云压低天际，远处灯塔发出微光，氛围张力强",
    "现代美术馆空旷大厅中的巨型悬浮雕塑，白墙与自然光极简干净",
    "古代战场遗址上的残破旗帜与石碑，夕阳斜照，厚重历史感",
    "未来厨房里的服务机器人正在制作甜点，色彩明快，材质细节真实",
    "安第斯山脉小村庄上空的热气球节，清晨冷暖光交错，层次分明",
    "神秘沙丘间半埋的古代机械遗迹，黄金沙粒纹理清晰，概念设计图",
    "城市高楼顶部的秘密泳池派对，蓝紫霓虹灯映在水面，时尚海报风",
    "晨雾森林中的猎人小屋，烟囱升起细烟，光线自然，真实质感",
    "星际旅行者的私人舱室，窗外是彩色星云，构图完整，氛围静谧",
    "小镇火车站台边的橙色猫咪与旧行李箱，温暖写实，情绪感强",
    "废弃工厂内长满植物的中庭，阳光从破碎天窗洒落，末世与新生并存",
    "北欧风格木屋客厅内的壁炉场景，落地窗外大雪纷飞，适合居家主题海报",
    "古希腊神殿遗址漂浮在云海之上，光束穿透云层，庄严神圣",
    "未来赛车在沙漠盐湖上高速掠过，地面倒影强烈，动态感十足",
    "夜晚露营地的篝火旁，几只柯基围坐看星星，轻松可爱，画面纯净",
    "高科技温室农场里层层叠叠的蔬菜塔，绿色光照，未来农业主题",
    "深夜便利店门口的机车骑士，雨水顺着头盔滴落，日式电影氛围",
    "峡湾边的红色渔村与雪山倒影，空气清冽，超写实风景摄影",
    "魔法学院天台的观星课堂，漂浮书页环绕学生，奇幻设定图",
    "未来水下酒店套房，窗外珊瑚与海龟缓缓游过，奢华而宁静",
    "黄昏草原上奔跑的黑马群，逆光扬起尘土，史诗感构图",
    "复古美式汽车旅馆的粉色霓虹招牌，棕榈树剪影映衬天空，怀旧商业摄影",
    "机械蜂群围绕透明蜂巢核心工作的未来实验装置，科技感强",
    "中式园林冬雪后的回廊与红梅，白雪映衬朱红建筑，构图优雅",
    "巨型树屋城市建立在参天古树之间，索桥连接各个居所，梦幻世界观",
    "岩浆洞穴内的晶体祭坛，橙红反光与紫色矿石形成强烈对比",
    "都市写字楼天台上的晨跑者，身后是金色日出与玻璃幕墙倒影",
    "极简白色展厅中的一把透明椅子，柔和阴影，产品广告级渲染",
    "热闹夜市中的糖画摊与蒸汽小吃车，颜色丰富但层次清晰",
    "外星沙漠上的双月夜空与孤独探路机器人，科幻感十足",
    "海边岩洞中的天然蓝色冰层，洞口透入清晨曙光，冷暖对比明显",
    "复古唱片店里认真挑选黑胶的女孩，木质货架与暖灯光营造氛围",
    "峡谷玻璃栈道上的游客与下方云海，广角透视，空间纵深强烈",
    "蒸汽火车穿越冬季松树林，白烟翻涌，复古海报风格",
    "未来城市中的垂直农田摩天楼，绿色植被覆盖建筑外立面",
    "巨浪拍打悬崖灯塔的瞬间，海鸟掠过，力量感突出",
    "古风药铺内部陈设整齐，药柜、铜秤与干草药细节丰富",
    "日落时分的机场跑道与停靠客机，暖金色天空映照机身金属反光",
    "悬浮在深空中的巨大环形城市，内部灯火如星河，科幻设定图",
    "森林溪流边搭建的木制露台餐桌，野花与灯串点缀，氛围轻松",
    "未来图书馆中的全息书架与悬空阅读平台，蓝白色极简科技风",
    "雨后的欧洲老城广场，石板地反射天空，街边咖啡座安静温暖",
    "机械巨龙盘踞在废墟王座之上，金属鳞片细节锐利，史诗奇幻风",
    "山谷中的薰衣草农场与白色风车，紫色花海层次丰富，适合壁纸",
    "火星殖民地温室外的红色尘暴逼近，人物穿着宇航服站在前景",
    "午夜办公室里唯一亮着灯的工位，电脑屏幕反光映出侧脸，都市叙事感",
    "古埃及风格地下神庙通道，两侧壁画在火把照耀下闪烁金色纹理",
    "海岸悬崖上的纯白婚礼场地，蓝海与白花布置营造浪漫氛围",
    "未来警用无人机编队穿越高楼之间，速度感强，都市科幻海报风",
    "木星轨道空间站里的巨型观测窗，远处行星纹理壮观清晰",
    "秋日枫树林中的日式温泉旅馆，暖灯从纸窗透出，宁静治愈",
    "黑夜沙滩上的荧光海浪与站立的人影，梦幻而神秘，画面纯净",
    "复古钟表匠工作台，齿轮与怀表零件铺满桌面，微距细节丰富",
    "未来体育馆中央的机甲格斗擂台，观众席灯光如星海，热血视觉冲击",
    "云端之上的玻璃步道花园，白色花朵被风吹动，构图轻盈通透",
    "废土公路旁的加油站遗迹，被风沙侵蚀，电影级末世氛围",
    "中式神话中的鲲在云海中翻涌，远方宫阙若隐若现，宏大史诗感",
    "清晨渔市里的冰台海鲜与忙碌摊主，生活气息浓厚，真实摄影风格",
    "太空货运仓库内部，机械叉车运送发光箱体，工业细节密集",
    "高山寺庙前覆盖薄雪的石阶，几盏灯笼映出柔和暖光，禅意宁静",
    "热带岛屿上的透明海水与水上木屋，正午强光下色彩通透鲜亮",
    "赛博寺庙内的僧侣与全息经文墙，传统与未来结合，视觉新鲜",
    "夜晚游乐园停运后的旋转木马，灯光未熄，轻微怀旧与孤独感",
    "未来植物实验室里的巨型捕蝇草样本，玻璃容器与冷色实验光突出细节",
    "老式电影院门口的霓虹招牌与排队人群，雨后地面反光，胶片叙事感",
    "极地海面上破冰船穿行于浮冰之间，天空被极光染成绿色，气势宏大",
    "中国西南梯田清晨灌水后的镜面倒影，人物点缀其中，层次鲜明",
    "宇宙港候机大厅中的旅人、行李机器人和飞船时刻屏，未来日常感",
    "河畔古城夜宴场景，灯火通明，小船穿梭，盛世国风画卷感",
    "风暴中的悬空岛屿群，闪电照亮岩壁与瀑布，奇幻电影海报风",
    "清爽现代书房里的大窗景观，桌面摆放简洁，适合家居封面图",
    "深夜停车场中的复古跑车与霓虹灯牌，色块分明，时尚广告风",
    "失落文明的巨大圆形石门立于草原中央，云影掠过，神秘探索感",
    "未来儿童乐园中的软体机器人与发光滑梯，色彩鲜活，安全友好",
    "葡萄园山坡上的石头酒庄，夕阳下大片葡萄藤延展至远方",
    "海底隧道餐厅中的双人餐桌，窗外鱼群环绕，浪漫而高级",
    "机甲维修车间内的工程师站在半拆解机体前，火花飞溅，工业感强",
    "山顶云海中的观景列车驶出隧道，红色车身与冷色环境形成对比",
    "中世纪石桥边的盔甲骑士与白马，清晨薄雾营造史诗冒险开场",
    "未来家居样板间中漂浮式家具与智能灯光系统，极简高级渲染",
    "夜晚沙漠音乐节主舞台，激光穿透尘雾，人群剪影与星空同框",
    "火山岛悬崖上的黑色城堡，海浪拍打岩壁，强对比奇幻视觉",
    "现代城市中央公园的晨练场景，雾气、阳光与树影层次丰富",
    "古代铸剑坊内部，火炉通红，铁匠挥锤瞬间火星四溅，力量感强",
    "未来高速磁悬浮列车驶过冰川峡谷，冷色画面中红色灯带形成焦点",
    "雨林树冠层上的研究平台与悬挂步道，绿色层次丰富，探险氛围浓厚",
    "大海中央孤独漂浮的小木屋与一盏亮灯窗户，极简叙事风格",
    "神秘月下湖泊边的银色独角兽，水面倒映星空，梦幻纯净",
    "复古照相馆里排列整齐的相机与暖黄灯泡，静物细节商业摄影风",
    "太空考古队在古代外星遗迹前展开扫描设备，科幻考古主题鲜明",
    "老城区楼顶晾衣绳与鸽群，傍晚夕光温暖，生活化电影镜头感",
    "玻璃穹顶下的热带植物园婚礼仪式，白绿配色纯净优雅",
    "未来深海采矿平台与远处巨型机械章鱼，工业科幻与危险感并存",
]

PROMPT_VARIANTS: List[str] = [
    "强调主体清晰与高对比",
    "加入体积光和环境氛围",
    "适合高质量壁纸输出",
    "保留商业海报级构图",
    "增强材质与纹理细节",
    "画面更干净，避免杂乱元素",
    "突出空间纵深和远近层次",
    "色彩统一，避免低质感",
]


class ImageLoadTestService:
    """管理后台触发的图片并发自测任务。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._job: Optional[Dict[str, Any]] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._image_models: List[str] = sorted(
            [
                str(model_id)
                for model_id, meta in MODEL_CONFIG.items()
                if isinstance(meta, dict) and str(meta.get("type") or "").strip().lower() == "image"
            ]
        )

    async def start_job(
        self,
        *,
        model: str,
        total_requests: int,
        duration_seconds: int,
        max_concurrency: int,
        timeout_seconds: int,
        prompt_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            if self._job and self._job.get("status") in {"queued", "running", "stopping"}:
                raise ValueError("当前已有图片并发自测任务在运行，请先等待完成或停止")

            now = datetime.now().isoformat(timespec="seconds")
            job_id = uuid.uuid4().hex[:12]
            prompt_prefix = str(prompt_prefix or "").strip()
            interval_seconds = duration_seconds / max(total_requests, 1)
            self._job = {
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "model": str(model or "random").strip() or "random",
                "total_requests": int(total_requests),
                "duration_seconds": int(duration_seconds),
                "max_concurrency": int(max_concurrency),
                "timeout_seconds": int(timeout_seconds),
                "prompt_prefix": prompt_prefix,
                "target_rps": round(total_requests / max(duration_seconds, 1), 2),
                "launch_interval_ms": int(interval_seconds * 1000),
                "launched": 0,
                "completed": 0,
                "succeeded": 0,
                "failed": 0,
                "in_flight": 0,
                "cancel_requested": False,
                "recent_errors": [],
                "recent_prompts": [],
                "duration_samples_ms": [],
                "last_error": None,
                "summary": "任务已创建，等待启动",
            }
            self._runner_task = asyncio.create_task(self._run_job(job_id))
            return self._build_snapshot_locked()

    async def stop_job(self) -> Dict[str, Any]:
        async with self._lock:
            if not self._job:
                return {"running": False, "job": None}
            if self._job.get("status") in {"completed", "failed", "cancelled"}:
                return {"running": False, "job": self._build_snapshot_locked()}
            self._job["cancel_requested"] = True
            if self._job.get("status") in {"queued", "running"}:
                self._job["status"] = "stopping"
                self._job["summary"] = "已请求停止，等待已发起请求完成"
            return {"running": True, "job": self._build_snapshot_locked()}

    async def get_status(self) -> Dict[str, Any]:
        async with self._lock:
            if not self._job:
                return {"running": False, "job": None}
            status = str(self._job.get("status") or "")
            return {
                "running": status in {"queued", "running", "stopping"},
                "job": self._build_snapshot_locked(),
            }

    async def _run_job(self, job_id: str) -> None:
        semaphore: Optional[asyncio.Semaphore] = None
        request_tasks: List[asyncio.Task] = []
        try:
            async with self._lock:
                if not self._job or self._job.get("job_id") != job_id:
                    return
                self._job["status"] = "running"
                self._job["started_at"] = datetime.now().isoformat(timespec="seconds")
                self._job["summary"] = "任务运行中，开始按节奏发起请求"
                total_requests = int(self._job.get("total_requests") or 0)
                duration_seconds = int(self._job.get("duration_seconds") or 0)
                max_concurrency = int(self._job.get("max_concurrency") or 1)
                model = str(self._job.get("model") or "")
                timeout_seconds = int(self._job.get("timeout_seconds") or 180)
                prompt_prefix = str(self._job.get("prompt_prefix") or "")

            semaphore = asyncio.Semaphore(max(1, max_concurrency))
            start_mono = time.perf_counter()
            interval_seconds = duration_seconds / max(total_requests, 1)

            for index in range(total_requests):
                if await self._is_cancel_requested(job_id):
                    break
                launch_at = start_mono + index * interval_seconds
                sleep_seconds = launch_at - time.perf_counter()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                if await self._is_cancel_requested(job_id):
                    break

                prompt = self._build_prompt(index=index, prompt_prefix=prompt_prefix)
                await self._record_launch(job_id, prompt)
                request_tasks.append(
                    asyncio.create_task(
                        self._run_single_request(
                            job_id=job_id,
                            model=model,
                            prompt=prompt,
                            timeout_seconds=timeout_seconds,
                            semaphore=semaphore,
                        )
                    )
                )

            if request_tasks:
                await asyncio.gather(*request_tasks, return_exceptions=True)

            async with self._lock:
                if not self._job or self._job.get("job_id") != job_id:
                    return
                cancelled = bool(self._job.get("cancel_requested"))
                self._job["status"] = "cancelled" if cancelled else "completed"
                self._job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                if cancelled:
                    self._job["summary"] = "任务已停止，未继续发起剩余请求"
                else:
                    self._job["summary"] = "任务已完成，全部请求已结束"
        except Exception as exc:
            async with self._lock:
                if self._job and self._job.get("job_id") == job_id:
                    self._job["status"] = "failed"
                    self._job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                    self._job["last_error"] = str(exc)
                    self._job["summary"] = f"任务执行失败：{str(exc)}"
                    self._push_recent_error_locked(f"任务异常：{str(exc)}")
        finally:
            async with self._lock:
                if self._job and self._job.get("job_id") == job_id:
                    self._job["in_flight"] = max(0, int(self._job.get("in_flight") or 0))

    async def _run_single_request(
        self,
        *,
        job_id: str,
        model: str,
        prompt: str,
        timeout_seconds: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            started_at = time.perf_counter()
            await self._increase_inflight(job_id)
            success = False
            error_message: Optional[str] = None
            request_timeout_seconds = max(10, int(timeout_seconds))
            image_total_timeout = max(30, int(config.image_total_timeout or 120))
            effective_timeout_seconds = request_timeout_seconds
            if request_timeout_seconds >= image_total_timeout:
                effective_timeout_seconds = max(request_timeout_seconds, image_total_timeout + 5)

            try:
                from ..api.routes import _handle_chat_completion_request

                request_model = self._choose_request_model(model)
                request = ChatCompletionRequest(
                    model=request_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    _handle_chat_completion_request(request, allow_cluster_dispatch=True),
                    timeout=effective_timeout_seconds,
                )
                status_code, body_text = self._extract_response_payload(response)
                body_error_message = self._extract_error_message_from_text(body_text)
                success = status_code < 400 and not body_error_message
                if not success:
                    error_message = self._normalize_error_message(
                        error_message=body_error_message or body_text,
                        status_code=status_code,
                        body_text=body_text,
                        timeout_seconds=effective_timeout_seconds,
                    )
            except HTTPException as exc:
                error_message = self._normalize_error_message(
                    error_message=exc.detail,
                    status_code=int(getattr(exc, "status_code", 500) or 500),
                    timeout_seconds=effective_timeout_seconds,
                    exc=exc,
                )
            except asyncio.TimeoutError:
                error_message = self._normalize_error_message(
                    error_message=None,
                    timeout_seconds=effective_timeout_seconds,
                    exc=asyncio.TimeoutError(),
                )
            except Exception as exc:
                error_message = self._normalize_error_message(
                    error_message=str(exc),
                    timeout_seconds=effective_timeout_seconds,
                    exc=exc,
                )
            finally:
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                await self._complete_request(
                    job_id=job_id,
                    success=success,
                    duration_ms=duration_ms,
                    error_message=error_message,
                )

    async def _record_launch(self, job_id: str, prompt: str) -> None:
        async with self._lock:
            if not self._job or self._job.get("job_id") != job_id:
                return
            self._job["launched"] = int(self._job.get("launched") or 0) + 1
            prompts = self._job.setdefault("recent_prompts", [])
            prompts.insert(0, prompt)
            del prompts[8:]
            launched = int(self._job.get("launched") or 0)
            total = int(self._job.get("total_requests") or 0)
            self._job["summary"] = f"已发起 {launched}/{total} 个请求"

    async def _increase_inflight(self, job_id: str) -> None:
        async with self._lock:
            if not self._job or self._job.get("job_id") != job_id:
                return
            self._job["in_flight"] = int(self._job.get("in_flight") or 0) + 1

    async def _complete_request(
        self,
        *,
        job_id: str,
        success: bool,
        duration_ms: int,
        error_message: Optional[str],
    ) -> None:
        async with self._lock:
            if not self._job or self._job.get("job_id") != job_id:
                return

            self._job["in_flight"] = max(0, int(self._job.get("in_flight") or 0) - 1)
            self._job["completed"] = int(self._job.get("completed") or 0) + 1
            if success:
                self._job["succeeded"] = int(self._job.get("succeeded") or 0) + 1
            else:
                self._job["failed"] = int(self._job.get("failed") or 0) + 1
                self._job["last_error"] = error_message
                self._push_recent_error_locked(error_message or "未知错误")

            samples = self._job.setdefault("duration_samples_ms", [])
            samples.append(int(duration_ms))
            if len(samples) > 400:
                del samples[:-400]

            completed = int(self._job.get("completed") or 0)
            total = int(self._job.get("total_requests") or 0)
            self._job["summary"] = f"已完成 {completed}/{total} 个请求"

    async def _is_cancel_requested(self, job_id: str) -> bool:
        async with self._lock:
            return bool(self._job and self._job.get("job_id") == job_id and self._job.get("cancel_requested"))

    def _build_snapshot_locked(self) -> Dict[str, Any]:
        job = dict(self._job or {})
        if not job:
            return {}

        samples = [int(item) for item in job.get("duration_samples_ms") or [] if isinstance(item, (int, float))]
        started_at = job.get("started_at")
        finished_at = job.get("finished_at")
        elapsed_seconds = self._calculate_elapsed_seconds(started_at, finished_at)
        completed = int(job.get("completed") or 0)
        launched = int(job.get("launched") or 0)
        total = max(1, int(job.get("total_requests") or 0))
        succeeded = int(job.get("succeeded") or 0)
        failed = int(job.get("failed") or 0)

        return {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "model": job.get("model"),
            "total_requests": total,
            "duration_seconds": int(job.get("duration_seconds") or 0),
            "max_concurrency": int(job.get("max_concurrency") or 0),
            "timeout_seconds": int(job.get("timeout_seconds") or 0),
            "prompt_prefix": job.get("prompt_prefix") or "",
            "target_rps": job.get("target_rps") or 0,
            "launch_interval_ms": int(job.get("launch_interval_ms") or 0),
            "launched": launched,
            "completed": completed,
            "succeeded": succeeded,
            "failed": failed,
            "in_flight": int(job.get("in_flight") or 0),
            "cancel_requested": bool(job.get("cancel_requested")),
            "launch_progress_percent": round(launched / total * 100, 1),
            "complete_progress_percent": round(completed / total * 100, 1),
            "success_rate": round((succeeded / completed) * 100, 1) if completed else 0.0,
            "avg_duration_ms": round(sum(samples) / len(samples), 1) if samples else 0.0,
            "p95_duration_ms": self._percentile(samples, 95),
            "max_duration_ms": max(samples) if samples else 0,
            "recent_errors": list(job.get("recent_errors") or []),
            "recent_prompts": list(job.get("recent_prompts") or []),
            "last_error": job.get("last_error"),
            "summary": job.get("summary") or "",
        }

    def _choose_request_model(self, configured_model: str) -> str:
        normalized = str(configured_model or "").strip()
        if normalized and normalized.lower() not in {"random", "随机"}:
            return normalized
        if not self._image_models:
            raise HTTPException(status_code=500, detail="没有可用的图片模型用于并发自测")
        return random.choice(self._image_models)

    def _push_recent_error_locked(self, message: str) -> None:
        errors = self._job.setdefault("recent_errors", []) if self._job else []
        errors.insert(0, str(message or "未知错误"))
        del errors[8:]

    @staticmethod
    def _extract_error_message_from_text(body_text: Any) -> str:
        raw_text = str(body_text or "").strip()
        if not raw_text:
            return ""
        try:
            payload = json.loads(raw_text)
        except Exception:
            return raw_text

        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = str(error_obj.get("message") or error_obj.get("detail") or "").strip()
                if message:
                    return message
            message = str(payload.get("detail") or payload.get("message") or "").strip()
            if message:
                return message
        return raw_text

    @classmethod
    def _normalize_error_message(
        cls,
        *,
        error_message: Any,
        timeout_seconds: int,
        status_code: Optional[int] = None,
        body_text: Any = None,
        exc: Optional[BaseException] = None,
    ) -> str:
        message = cls._extract_error_message_from_text(error_message)
        if not message and body_text is not None:
            message = cls._extract_error_message_from_text(body_text)
        if message:
            return message

        exc_obj = exc
        exc_name = exc_obj.__class__.__name__ if exc_obj else ""
        image_total_timeout = max(30, int(config.image_total_timeout or 120))

        if isinstance(exc_obj, asyncio.TimeoutError) or exc_name == "TimeoutError":
            return f"请求超时（>{int(timeout_seconds)} 秒，可能与图片生成总超时 {image_total_timeout} 秒有关）"

        if isinstance(exc_obj, asyncio.CancelledError) or exc_name == "CancelledError":
            return f"请求被取消（可能由图片生成总超时 {image_total_timeout} 秒触发）"

        if isinstance(exc_obj, HTTPException):
            detail = cls._extract_error_message_from_text(getattr(exc_obj, "detail", ""))
            if detail:
                return detail

        if status_code and int(status_code) >= 400:
            return f"HTTP {int(status_code)} 错误（响应体为空）"

        if exc_name:
            return f"{exc_name}（异常消息为空）"
        return "未知错误（异常消息为空，可能是超时或取消触发）"

    @staticmethod
    def _extract_response_payload(response: Any) -> tuple[int, str]:
        if isinstance(response, Response):
            body = getattr(response, "body", b"") or b""
            if isinstance(body, bytes):
                text = body.decode("utf-8", errors="ignore")
            else:
                text = str(body)
            return int(getattr(response, "status_code", 200) or 200), text[:500]

        if isinstance(response, dict):
            return 200, json.dumps(response, ensure_ascii=False)[:500]

        return 200, str(response)[:500]

    @staticmethod
    def _calculate_elapsed_seconds(started_at: Optional[str], finished_at: Optional[str]) -> int:
        if not started_at:
            return 0
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(finished_at) if finished_at else datetime.now()
            return max(0, int((end_dt - start_dt).total_seconds()))
        except Exception:
            return 0

    @staticmethod
    def _percentile(samples: List[int], percentile: int) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        if len(ordered) == 1:
            return float(ordered[0])
        index = max(0, min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))))
        return float(ordered[index])

    @staticmethod
    def _build_prompt(*, index: int, prompt_prefix: str) -> str:
        prompt_seed = PROMPT_LIBRARY[index % len(PROMPT_LIBRARY)]
        cycle = index // len(PROMPT_LIBRARY)
        extra = ""
        if cycle > 0:
            extra = f"，额外变体方向：{PROMPT_VARIANTS[cycle % len(PROMPT_VARIANTS)]}"
        base_prompt = f"{prompt_seed}{extra}。第 {index + 1} 次并发自测样本。"
        prefix = str(prompt_prefix or "").strip()
        return f"{prefix}，{base_prompt}" if prefix else base_prompt


image_load_test_service = ImageLoadTestService()