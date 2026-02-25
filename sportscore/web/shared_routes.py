"""Shared API routes for the unified sportscore web app.

All routes delegate to ``g.services`` (a ``SportServices`` dataclass populated
per-request by the sport plugin's ``get_web_services``).  No sport-specific
imports appear here.
"""

import logging
import os
import threading
from datetime import datetime

from bson import ObjectId
from flask import g, jsonify, request, send_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_config(config):
    """Make a Mongo doc JSON-safe (ObjectId, datetime)."""
    if not config:
        return {}
    c = dict(config)
    if "_id" in c:
        c["_id"] = str(c["_id"])
    for k in ("created_at", "updated_at", "trained_at", "artifacts_saved_at"):
        if c.get(k):
            c[k] = c[k].isoformat() if hasattr(c[k], "isoformat") else str(c[k])
    return c


def _parse_year_list(val):
    if val is None:
        return None
    if isinstance(val, list):
        return [int(y) for y in val]
    if isinstance(val, str):
        return [int(y.strip()) for y in val.split(",") if y.strip()]
    return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_shared_routes(app):
    """Attach all shared API routes to *app*."""

    # ==================================================================
    # Model Config CRUD
    # ==================================================================

    @app.route("/<league_id>/api/model-config/save", methods=["POST"])
    def api_model_config_save(league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            if data.get("ensemble"):
                return jsonify(success=True, message="Ensembles saved during training")

            c_supported = set(g.services.model.c_supported_models or [])
            features = data.get("features") or []
            model_types = data.get("model_types") or g.services.model.default_model_types or ["LogisticRegression"]

            saved = []
            for idx, mt in enumerate(model_types):
                c_value = None
                if mt in c_supported and data.get("c_values"):
                    try:
                        c_value = float(data["c_values"][0])
                    except Exception:
                        c_value = data["c_values"][0]

                config_id, cfg = config_manager.create_classifier_config(
                    model_type=mt,
                    features=sorted(features),
                    c_value=c_value,
                    use_time_calibration=data.get("use_time_calibration", False),
                    calibration_method=data.get("calibration_method"),
                    begin_year=data.get("begin_year"),
                    calibration_years=data.get("calibration_years"),
                    evaluation_year=data.get("evaluation_year"),
                    min_games_played=data.get("min_games_played", 15),
                    exclude_seasons=data.get("exclude_seasons"),
                    selected=(idx == 0),
                    name=data.get("name"),
                )
                saved.append({"model_type": mt, "config_id": config_id})

            return jsonify(success=True, saved_configs=saved)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-config/load", methods=["GET"])
    def api_model_config_load(league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_selected_config()
            if not doc:
                return jsonify(success=False, error="No config found")
            config = {
                "model_types": [doc.get("model_type")],
                "features": doc.get("features", []),
                "c_values": [doc.get("best_c_value")] if doc.get("best_c_value") is not None else [],
                "use_master": doc.get("use_master", True),
                "use_time_calibration": doc.get("use_time_calibration", False),
                "calibration_method": doc.get("calibration_method"),
                "begin_year": doc.get("begin_year"),
                "calibration_years": doc.get("calibration_years"),
                "evaluation_year": doc.get("evaluation_year"),
                "min_games_played": doc.get("min_games_played", 15),
                "point_model_id": doc.get("point_model_id"),
            }
            return jsonify(success=True, config=config)
        except Exception as e:
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs", methods=["GET"])
    def api_model_configs(league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            configs = config_manager.list_configs()

            # Enforce exactly one selected
            selected = [c for c in configs if c.get("selected")]
            if configs and not selected:
                config_manager.set_selected_config(configs[0]["_id"])
                configs[0]["selected"] = True
            elif len(selected) > 1:
                keep = max(selected, key=lambda c: (
                    c.get("trained_at") or datetime.min,
                    c.get("updated_at") or datetime.min,
                ))
                config_manager.set_selected_config(str(keep["_id"]))
                for c in configs:
                    if c.get("selected") and str(c["_id"]) != str(keep["_id"]):
                        c["selected"] = False

            return jsonify(success=True, configs=[_serialize_config(c) for c in configs])
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/selected", methods=["GET"])
    def api_model_configs_selected(league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_selected_config()
            if not doc:
                return jsonify(success=False, error="No selected config"), 404
            return jsonify(success=True, config=_serialize_config(doc))
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/<config_id>", methods=["GET"])
    def api_model_config_get(config_id, league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_config(config_id)
            if not doc:
                return jsonify(success=False, error="Config not found"), 404
            return jsonify(success=True, config=_serialize_config(doc))
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/<config_id>", methods=["PUT"])
    def api_model_config_update(config_id, league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            if not data:
                return jsonify(success=False, error="No data"), 400

            doc = config_manager.get_config(config_id)
            if not doc:
                return jsonify(success=False, error="Config not found"), 404

            # Handle selection toggle
            if "selected" in data and data["selected"]:
                config_manager.set_selected_config(config_id)

            has_update = False
            if "name" in data:
                doc["name"] = data["name"]; has_update = True
            if "selected" in data:
                doc["selected"] = bool(data["selected"]); has_update = True
            if "model_type" in data:
                doc["model_type"] = data["model_type"]; has_update = True
            if "features" in data:
                feats = sorted(data["features"])
                doc["features"] = feats
                doc["feature_count"] = len(feats)
                if hasattr(config_manager, "generate_feature_set_hash"):
                    doc["feature_set_hash"] = config_manager.generate_feature_set_hash(feats)
                has_update = True
            if "best_c_value" in data:
                doc["best_c_value"] = data["best_c_value"]; has_update = True
            elif "c_values" in data and data["c_values"]:
                try:
                    doc["best_c_value"] = float(data["c_values"][0])
                except Exception:
                    doc["best_c_value"] = data["c_values"][0]
                has_update = True
            for key in ("use_time_calibration", "calibration_method", "begin_year",
                        "evaluation_year", "min_games_played"):
                if key in data:
                    val = data[key]
                    if key in ("begin_year", "evaluation_year", "min_games_played") and val is not None:
                        val = int(val)
                    elif key == "use_time_calibration":
                        val = bool(val)
                    doc[key] = val; has_update = True
            if "calibration_years" in data:
                doc["calibration_years"] = _parse_year_list(data["calibration_years"]); has_update = True
            if "exclude_seasons" in data:
                excl = data["exclude_seasons"]
                if isinstance(excl, str):
                    excl = [int(y.strip()) for y in excl.split(",") if y.strip()] if excl.strip() else None
                elif isinstance(excl, list):
                    excl = [int(y) for y in excl] if excl else None
                doc["exclude_seasons"] = excl; has_update = True

            if not has_update:
                return jsonify(success=False, error="No valid fields"), 400

            config_manager.save_config(doc)
            return jsonify(success=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/<config_id>", methods=["DELETE"])
    def api_model_config_delete(config_id, league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_config(config_id)
            if not doc:
                return jsonify(success=False, error="Config not found"), 404
            was_selected = doc.get("selected", False)
            config_manager.delete_config(config_id)
            if was_selected:
                configs = config_manager.list_configs()
                if configs:
                    config_manager.set_selected_config(configs[0]["_id"])
            return jsonify(success=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/<config_id>/clone", methods=["POST"])
    def api_model_config_clone(config_id, league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_config(config_id)
            if not doc:
                return jsonify(success=False, error="Config not found"), 404

            doc.pop("_id", None)
            doc["name"] = (doc.get("name") or "config") + " (copy)"
            doc["selected"] = False
            # Strip training artifacts so the clone is untrained
            for key in ("trained_at", "metrics", "model_path", "training_csv",
                        "accuracy", "brier_score", "log_loss", "roc_auc"):
                doc.pop(key, None)
            doc["created_at"] = datetime.utcnow()

            config_manager.save_config(doc)
            new_id = str(doc.get("_id", ""))
            return jsonify(success=True, config_id=new_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/<config_id>/download", methods=["GET"])
    def api_model_config_download(config_id, league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_config(config_id)
            if not doc:
                return jsonify(success=False, error="Config not found"), 404
            csv_path = doc.get("training_csv")
            if not csv_path or not os.path.exists(csv_path):
                return jsonify(success=False, error="Training CSV not found"), 404
            return send_file(csv_path, as_attachment=True, download_name=os.path.basename(csv_path))
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    # ==================================================================
    # Training
    # ==================================================================

    @app.route("/<league_id>/api/model-config/train", methods=["POST"])
    def api_model_config_train(league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Training not available"), 501
        try:
            data = request.json or {}
            league = g.league
            jobs_svc = g.services.jobs

            # Ensemble training
            if data.get("ensemble"):
                return _train_ensemble(data, league, training_service, jobs_svc)

            # Config-id mode: retrain a saved config by its _id
            config_id = data.get("config_id")
            if config_id and not data.get("model_types"):
                job_id = jobs_svc.create(job_type="train", league=league) if jobs_svc else str(ObjectId())

                def _run(job_id, config_id, league):
                    try:
                        training_service.train_by_config_id(
                            config_id, job_id=job_id, league=league,
                        )
                        if jobs_svc:
                            jobs_svc.complete(job_id, "Training completed.", league=league)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        if jobs_svc:
                            jobs_svc.fail(job_id, str(e), league=league)

                threading.Thread(target=_run, args=(job_id, config_id, league), daemon=True).start()
                return jsonify(success=True, job_id=job_id)

            # Legacy grid mode
            model_types = data.get("model_types") or g.services.model.default_model_types or ["LogisticRegression"]
            c_values = [float(x) for x in (data.get("c_values") or ["0.1"])]
            features = data.get("features") or []

            job_id = jobs_svc.create(job_type="train", league=league) if jobs_svc else str(ObjectId())

            def _run_grid(job_id, model_types, c_values, features, data, league):
                try:
                    def progress_cb(pct, msg):
                        if jobs_svc:
                            jobs_svc.update(job_id, pct, msg, league=league)

                    training_service.train_model_grid(
                        model_types=model_types,
                        c_values=c_values,
                        features=features,
                        use_time_calibration=data.get("use_time_calibration", False),
                        calibration_method=data.get("calibration_method", "isotonic"),
                        begin_year=data.get("begin_year"),
                        calibration_years=data.get("calibration_years"),
                        evaluation_year=data.get("evaluation_year"),
                        min_games_played=data.get("min_games_played", 15),
                        exclude_seasons=data.get("exclude_seasons"),
                        use_master=data.get("use_master", True),
                        progress_callback=progress_cb,
                    )
                    if jobs_svc:
                        jobs_svc.complete(job_id, "Training completed.", league=league)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    if jobs_svc:
                        jobs_svc.fail(job_id, str(e), league=league)

            threading.Thread(
                target=_run_grid,
                args=(job_id, model_types, c_values, features, data, league),
                daemon=True,
            ).start()
            return jsonify(success=True, job_id=job_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    def _train_ensemble(data, league, training_service, jobs_svc):
        """Handle ensemble training request."""
        base_ids = [str(m) for m in data.get("ensemble_models", [])]
        meta_model_type = (data.get("model_types") or ["LogisticRegression"])[0]
        meta_c_value = float((data.get("c_values") or [0.1])[0])
        stacking_mode = data.get("stacking_mode") or "naive"
        ensemble_id = data.get("ensemble_id")
        name = data.get("name")

        job_id = jobs_svc.create(job_type="train", league=league, config_id=ensemble_id) if jobs_svc else str(ObjectId())

        def _run(job_id, base_ids, meta_model_type, meta_c_value, stacking_mode, data, league, name, ens_id):
            try:
                training_service.train_ensemble(
                    meta_model_type=meta_model_type,
                    base_model_names_or_ids=base_ids,
                    meta_c_value=meta_c_value,
                    extra_features=data.get("ensemble_meta_features") or None,
                    stacking_mode=stacking_mode,
                    use_disagree=data.get("ensemble_use_disagree", False),
                    use_conf=data.get("ensemble_use_conf", False),
                    use_logit=data.get("ensemble_use_logit", False),
                    ensemble_id=ens_id,
                    name=name,
                    meta_calibration_method=data.get("meta_calibration_method"),
                    meta_train_years=_parse_year_list(data.get("meta_train_years")),
                    meta_calibration_years=_parse_year_list(data.get("meta_calibration_years")),
                    meta_evaluation_year=int(data["meta_evaluation_year"]) if data.get("meta_evaluation_year") else None,
                )
                if jobs_svc:
                    jobs_svc.complete(job_id, "Ensemble training completed.", league=league)
            except Exception as e:
                import traceback; traceback.print_exc()
                if jobs_svc:
                    jobs_svc.fail(job_id, str(e), league=league)

        threading.Thread(
            target=_run,
            args=(job_id, base_ids, meta_model_type, meta_c_value, stacking_mode, data, league, name, ensemble_id),
            daemon=True,
        ).start()
        return jsonify(success=True, job_id=job_id)

    # ==================================================================
    # Ensemble API
    # ==================================================================

    @app.route("/<league_id>/api/ensembles", methods=["GET"])
    def api_ensembles(league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            ensembles = training_service.list_ensembles()
            return jsonify(success=True, ensembles=ensembles)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/available-base-models", methods=["GET"])
    def api_ensembles_available_base(league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            models = training_service.list_available_base_models()
            return jsonify(success=True, models=models)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/<ensemble_id>", methods=["GET"])
    def api_ensemble_get(ensemble_id, league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            doc = config_manager.get_config(ensemble_id)
            if not doc:
                return jsonify(success=False, error="Ensemble not found"), 404
            doc = _serialize_config(doc)
            if "ensemble_models" in doc:
                doc["ensemble_models"] = [str(m) for m in doc["ensemble_models"]]
            return jsonify(success=True, ensemble=doc)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/<ensemble_id>", methods=["PUT"])
    def api_ensemble_update(ensemble_id, league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            result = training_service.update_ensemble_settings(ensemble_id, data)
            return jsonify(result)
        except ValueError as e:
            return jsonify(success=False, error=str(e)), 404
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/validate", methods=["POST"])
    def api_ensembles_validate(league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            training_service.validate_ensemble_models(data.get("ensemble_models", []))
            return jsonify(success=True, message="Validation passed")
        except ValueError as e:
            return jsonify(success=False, error=str(e)), 400
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/<ensemble_id>/retrain-meta", methods=["POST"])
    def api_ensemble_retrain_meta(ensemble_id, league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            ensemble = config_manager.get_config(ensemble_id)
            if not ensemble:
                return jsonify(success=False, error="Ensemble not found"), 404

            data = request.json or {}
            # Merge request overrides with DB values
            ensemble_models = data.get("ensemble_models", ensemble.get("ensemble_models", []))
            meta_model_type = data.get("meta_model_type", ensemble.get("model_type", "LogisticRegression"))
            meta_c_value = data.get("meta_c_value") or data.get("best_c_value") or ensemble.get("best_c_value", 0.1)
            try:
                meta_c_value = float(meta_c_value)
            except (TypeError, ValueError):
                meta_c_value = 0.1
            stacking_mode = data.get("stacking_mode", ensemble.get("stacking_mode", "naive"))
            ensemble_name = data.get("name") or ensemble.get("name")
            ensemble_meta_features = data.get("ensemble_meta_features", ensemble.get("ensemble_meta_features", []))
            ensemble_use_disagree = data.get("ensemble_use_disagree", ensemble.get("ensemble_use_disagree", False))
            ensemble_use_conf = data.get("ensemble_use_conf", ensemble.get("ensemble_use_conf", False))
            ensemble_use_logit = data.get("ensemble_use_logit", ensemble.get("ensemble_use_logit", False))

            meta_cal_method = data.get("meta_calibration_method") or ensemble.get("meta_calibration_method")
            meta_train_years = _parse_year_list(data.get("meta_train_years") or ensemble.get("meta_train_years"))
            meta_cal_years = _parse_year_list(data.get("meta_calibration_years") or ensemble.get("meta_calibration_years"))
            meta_eval_year = data.get("meta_evaluation_year") or ensemble.get("meta_evaluation_year")
            try:
                meta_eval_year = int(meta_eval_year) if meta_eval_year else None
            except (TypeError, ValueError):
                meta_eval_year = None
            if not meta_cal_method:
                meta_cal_method = meta_train_years = meta_cal_years = meta_eval_year = None

            # Save settings
            settings = {
                "ensemble_models": ensemble_models,
                "ensemble_meta_features": ensemble_meta_features,
                "ensemble_use_disagree": ensemble_use_disagree,
                "ensemble_use_conf": ensemble_use_conf,
                "ensemble_use_logit": ensemble_use_logit,
                "model_type": meta_model_type,
                "best_c_value": meta_c_value,
                "stacking_mode": stacking_mode,
                "meta_calibration_method": meta_cal_method,
                "meta_train_years": meta_train_years,
                "meta_calibration_years": meta_cal_years,
                "meta_evaluation_year": meta_eval_year,
            }
            if ensemble_name:
                settings["name"] = ensemble_name
            training_service.update_ensemble_settings(ensemble_id, settings)

            league = g.league
            jobs_svc = g.services.jobs
            job_id = jobs_svc.create(job_type="train", league=league, config_id=ensemble_id) if jobs_svc else str(ObjectId())

            base_ids = [str(m) for m in ensemble_models]

            def _run():
                try:
                    training_service.train_ensemble(
                        meta_model_type=meta_model_type,
                        base_model_names_or_ids=base_ids,
                        meta_c_value=meta_c_value,
                        extra_features=ensemble_meta_features or None,
                        stacking_mode=stacking_mode,
                        use_disagree=ensemble_use_disagree,
                        use_conf=ensemble_use_conf,
                        use_logit=ensemble_use_logit,
                        name=ensemble_name,
                        ensemble_id=ensemble_id,
                        meta_calibration_method=meta_cal_method,
                        meta_train_years=meta_train_years,
                        meta_calibration_years=meta_cal_years,
                        meta_evaluation_year=meta_eval_year,
                    )
                    if jobs_svc:
                        jobs_svc.complete(job_id, "Meta-model retrained successfully", league=league)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    if jobs_svc:
                        jobs_svc.fail(job_id, str(e), league=league)

            threading.Thread(target=_run, daemon=True).start()
            return jsonify(success=True, job_id=job_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/<ensemble_id>/retrain-base/<base_model_id>", methods=["POST"])
    def api_ensemble_retrain_base(ensemble_id, base_model_id, league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            league = g.league
            jobs_svc = g.services.jobs
            job_id = jobs_svc.create(job_type="retrain_base", league=league) if jobs_svc else str(ObjectId())

            def _run():
                try:
                    def progress_cb(pct, msg):
                        if jobs_svc:
                            jobs_svc.update(job_id, pct, msg, league=league)
                    training_service.retrain_base_model(
                        ensemble_id=ensemble_id,
                        base_model_id=base_model_id,
                        retrain_meta=True,
                        progress_callback=progress_cb,
                    )
                    if jobs_svc:
                        jobs_svc.complete(job_id, "Base model retrained", league=league)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    if jobs_svc:
                        jobs_svc.fail(job_id, str(e), league=league)

            threading.Thread(target=_run, daemon=True).start()
            return jsonify(success=True, job_id=job_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/ensembles/<ensemble_id>/update-calibration", methods=["POST"])
    def api_ensemble_update_calibration(ensemble_id, league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            begin_year = data.get("begin_year")
            calibration_years = data.get("calibration_years")
            evaluation_year = data.get("evaluation_year")
            if not all([begin_year, calibration_years, evaluation_year]):
                return jsonify(success=False, error="begin_year, calibration_years, evaluation_year required"), 400

            league = g.league
            jobs_svc = g.services.jobs
            job_id = jobs_svc.create(job_type="recalibrate", league=league) if jobs_svc else str(ObjectId())

            def _run():
                try:
                    def progress_cb(pct, msg):
                        if jobs_svc:
                            jobs_svc.update(job_id, pct, msg, league=league)
                    training_service.recalibrate_ensemble(
                        ensemble_id=ensemble_id,
                        begin_year=begin_year,
                        calibration_years=calibration_years,
                        evaluation_year=evaluation_year,
                        calibration_method=data.get("calibration_method", "isotonic"),
                        exclude_seasons=data.get("exclude_seasons"),
                        min_games_played=data.get("min_games_played"),
                        progress_callback=progress_cb,
                    )
                    if jobs_svc:
                        jobs_svc.complete(job_id, "Recalibration complete", league=league)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    if jobs_svc:
                        jobs_svc.fail(job_id, str(e), league=league)

            threading.Thread(target=_run, daemon=True).start()
            return jsonify(success=True, job_id=job_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/model-configs/create-ensemble", methods=["POST"])
    def api_create_ensemble(league_id=None):
        training_service = g.services.model.training_service
        if training_service is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            meta_c_value = data.get("meta_c_value")
            if meta_c_value is not None:
                try:
                    meta_c_value = float(meta_c_value)
                except (TypeError, ValueError):
                    meta_c_value = 0.1
            result = training_service.create_ensemble_config(
                base_model_ids=data.get("ensemble_models", []),
                meta_model_type=data.get("meta_model_type", "LogisticRegression"),
                meta_c_value=meta_c_value,
                stacking_mode=data.get("stacking_mode", "naive"),
                ensemble_meta_features=data.get("ensemble_meta_features", []),
                ensemble_use_disagree=data.get("ensemble_use_disagree", False),
                ensemble_use_conf=data.get("ensemble_use_conf", False),
                ensemble_use_logit=data.get("ensemble_use_logit", False),
                name=data.get("name"),
            )
            return jsonify(success=True, ensemble_id=result["ensemble_id"], is_new=result["is_new"])
        except ValueError as e:
            return jsonify(success=False, error=str(e)), 400
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    # ==================================================================
    # Points Model API
    # ==================================================================

    @app.route("/<league_id>/api/points-model/configs", methods=["GET"])
    def api_points_model_configs(league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            configs = config_manager.list_points_configs()
            return jsonify(success=True, configs=[_serialize_config(c) for c in configs])
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/points-model/select-config", methods=["POST"])
    def api_points_model_select(league_id=None):
        config_manager = g.services.model.config_manager
        if config_manager is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            config_id = data.get("config_id")
            if not config_id:
                return jsonify(success=False, error="config_id required"), 400
            doc = config_manager.get_points_config(config_id)
            if not doc:
                return jsonify(success=False, error="Config not found"), 404
            is_selected = doc.get("selected", False)
            if is_selected:
                config_manager.deselect_all_points_configs()
            else:
                config_manager.set_selected_points_config(config_id)
            return jsonify(success=True, selected=not is_selected)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/points-model/train", methods=["POST"])
    def api_points_model_train(league_id=None):
        points_trainer = g.services.model.points_trainer
        if points_trainer is None:
            return jsonify(success=False, error="Points training not available"), 501
        try:
            data = request.json or {}
            league = g.league
            jobs_svc = g.services.jobs
            job_id = jobs_svc.create(job_type="train_points", league=league) if jobs_svc else str(ObjectId())

            def _run():
                try:
                    points_trainer.train_from_request(data, job_id=job_id, league=league)
                    if jobs_svc:
                        jobs_svc.complete(job_id, "Points model training completed.", league=league)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    if jobs_svc:
                        jobs_svc.fail(job_id, str(e), league=league)

            threading.Thread(target=_run, daemon=True).start()
            return jsonify(success=True, job_id=job_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    # ==================================================================
    # Job Management
    # ==================================================================

    @app.route("/<league_id>/api/jobs/<job_id>", methods=["GET"])
    def api_job_status(job_id, league_id=None):
        jobs_svc = g.services.jobs
        if jobs_svc is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            job = jobs_svc.get_job(job_id, league=g.league)
            if not job:
                return jsonify(success=False, error="Job not found"), 404
            return jsonify(success=True, job=_serialize_config(job))
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/jobs/running/<job_type>", methods=["GET"])
    def api_jobs_running(job_type, league_id=None):
        jobs_svc = g.services.jobs
        if jobs_svc is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            job = jobs_svc.get_running(job_type, league=g.league)
            if not job:
                return jsonify(success=True, job=None)
            return jsonify(success=True, job=_serialize_config(job))
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    # ==================================================================
    # Master Training
    # ==================================================================

    @app.route("/<league_id>/api/master-training/columns", methods=["GET"])
    def api_master_training_columns(league_id=None):
        try:
            import pandas as pd
            csv_path = g.services.features.master_training_csv
            if not csv_path or not os.path.exists(csv_path):
                return jsonify(error="Master training CSV not found"), 404
            df = pd.read_csv(csv_path, nrows=0)
            columns = list(df.columns)
            total_rows = sum(1 for _ in open(csv_path)) - 1  # subtract header
            return jsonify(columns=columns, total_rows=total_rows)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route("/<league_id>/api/master-training/rows", methods=["GET"])
    def api_master_training_rows(league_id=None):
        try:
            import pandas as pd
            csv_path = g.services.features.master_training_csv
            if not csv_path or not os.path.exists(csv_path):
                return jsonify(error="Master training CSV not found"), 404

            offset = int(request.args.get("offset", 0))
            limit = int(request.args.get("limit", 100))
            columns_param = request.args.get("columns", "")
            sort_column = request.args.get("sort_column", "")
            sort_direction = request.args.get("sort_direction", "asc")

            # Date filters
            year_min = request.args.get("year_min")
            year_max = request.args.get("year_max")
            month_min = request.args.get("month_min")
            month_max = request.args.get("month_max")
            day_min = request.args.get("day_min")
            day_max = request.args.get("day_max")
            date_start = request.args.get("date_start")
            date_end = request.args.get("date_end")

            requested_columns = [c.strip() for c in columns_param.split(",") if c.strip()] if columns_param else []
            df = pd.read_csv(csv_path)

            # Apply date filters
            if "Year" in df.columns and "Month" in df.columns and "Day" in df.columns:
                for col_name, param_val, op in [
                    ("Year", year_min, ">="), ("Year", year_max, "<="),
                    ("Month", month_min, ">="), ("Month", month_max, "<="),
                    ("Day", day_min, ">="), ("Day", day_max, "<="),
                ]:
                    if param_val:
                        v = int(param_val)
                        if op == ">=":
                            df = df[df[col_name].astype(int) >= v]
                        else:
                            df = df[df[col_name].astype(int) <= v]
                if date_start or date_end:
                    df = df.copy()
                    df["_fd"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
                    if date_start:
                        df = df[df["_fd"] >= pd.to_datetime(date_start)]
                    if date_end:
                        df = df[df["_fd"] <= pd.to_datetime(date_end)]
                    df = df.drop("_fd", axis=1)

            if requested_columns:
                avail = [c for c in requested_columns if c in df.columns]
                if avail:
                    df = df[avail]
            if sort_column and sort_column in df.columns:
                df = df.sort_values(by=sort_column, ascending=(sort_direction == "asc"), na_position="last")

            total_count = len(df)
            paginated = df.iloc[offset: offset + limit]
            rows = paginated.to_dict("records")
            for row in rows:
                for k, v in row.items():
                    if pd.isna(v):
                        row[k] = None
            return jsonify(rows=rows, has_more=(offset + limit < total_count), total_count=total_count, offset=offset, limit=limit)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(error=str(e)), 500

    @app.route("/<league_id>/api/master-training/resolve-dependencies", methods=["POST"])
    def api_master_training_resolve_deps(league_id=None):
        resolver = g.services.features.dependency_resolver
        if resolver is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            result = resolver(data.get("feature_substrings", []), data.get("match_mode", "OR"))
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/regenerate-features", methods=["POST"])
    def api_master_training_regenerate_features(league_id=None):
        regenerator = g.services.features.regenerator
        if regenerator is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            if not data.get("confirmed"):
                return jsonify(success=False, error="User confirmation required"), 400
            result = regenerator(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/possible-features", methods=["GET"])
    def api_master_training_possible_features(league_id=None):
        getter = g.services.features.possible_getter
        if getter is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            features = getter()
            return jsonify(success=True, features=features)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/delete-column", methods=["POST"])
    def api_master_training_delete_column(league_id=None):
        try:
            import pandas as pd
            import shutil
            data = request.json or {}
            column_name = data.get("column_name")
            if not column_name:
                return jsonify(success=False, error="column_name required"), 400

            csv_path = g.services.features.master_training_csv
            if not csv_path or not os.path.exists(csv_path):
                return jsonify(success=False, error="Master training CSV not found"), 404

            metadata_cols = {"Year", "Month", "Day", "Home", "Away", "HomeWon"}
            if column_name in metadata_cols:
                return jsonify(success=False, error=f"Cannot delete metadata column: {column_name}"), 400

            df = pd.read_csv(csv_path)
            if column_name not in df.columns:
                return jsonify(success=False, error=f"Column not found: {column_name}"), 404
            df = df.drop(columns=[column_name])
            tmp = f"{csv_path}.tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            df.to_csv(tmp, index=False)
            shutil.move(tmp, csv_path)
            return jsonify(success=True, message=f"Column {column_name} deleted")
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/add-columns", methods=["POST"])
    def api_master_training_add_columns(league_id=None):
        adder = g.services.features.column_adder
        if adder is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            if not data.get("confirmed"):
                return jsonify(success=False, error="User confirmation required"), 400
            result = adder(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/available-seasons", methods=["GET"])
    def api_master_training_available_seasons(league_id=None):
        getter = g.services.features.available_seasons_getter
        if getter is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            seasons = getter()
            return jsonify(success=True, seasons=seasons)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/regenerate-seasons", methods=["POST"])
    def api_master_training_regenerate_seasons(league_id=None):
        regenerator = g.services.features.season_regenerator
        if regenerator is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            result = regenerator(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/master-training/regenerate-full", methods=["POST"])
    def api_master_training_regenerate_full(league_id=None):
        regenerator = g.services.features.full_regenerator
        if regenerator is None:
            return jsonify(success=False, error="Not available for this sport"), 501
        try:
            data = request.json or {}
            result = regenerator(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/download-master-training", methods=["GET"])
    def api_download_master_training(league_id=None):
        try:
            csv_path = g.services.features.master_training_csv
            if not csv_path or not os.path.exists(csv_path):
                return jsonify(success=False, error="Master training CSV not found"), 404
            return send_file(csv_path, as_attachment=True, download_name=os.path.basename(csv_path))
        except Exception as e:
            return jsonify(success=False, error=str(e)), 500

    # ==================================================================
    # Market API
    # ==================================================================

    @app.route("/<league_id>/api/market-prices", methods=["GET"])
    def api_market_prices(league_id=None):
        market_data_fn = g.services.market.prices_getter
        if market_data_fn is None:
            return jsonify(success=False, error="Market data not available"), 501
        try:
            date_str = request.args.get("date")
            if not date_str:
                return jsonify(success=False, error="Missing date parameter"), 400
            result = market_data_fn(date_str, g.db, g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/api/market/dashboard", methods=["GET"])
    @app.route("/<league_id>/api/market/dashboard", methods=["GET"])
    def api_market_dashboard(league_id=None):
        dashboard_fn = g.services.market.dashboard_getter
        if dashboard_fn is None:
            return jsonify(success=False, error="Market dashboard not available"), 501
        try:
            refresh = request.args.get("refresh") == "1"
            result = dashboard_fn(refresh=refresh)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/api/market/fills", methods=["GET"])
    @app.route("/<league_id>/api/market/fills", methods=["GET"])
    def api_market_fills(league_id=None):
        fills_fn = g.services.market.fills_getter
        if fills_fn is None:
            return jsonify(success=False, error="Market fills not available"), 501
        try:
            cursor = request.args.get("cursor")
            limit = int(request.args.get("limit", 100))
            result = fills_fn(cursor=cursor, limit=limit)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/api/market/settlements", methods=["GET"])
    @app.route("/<league_id>/api/market/settlements", methods=["GET"])
    def api_market_settlements(league_id=None):
        settlements_fn = g.services.market.settlements_getter
        if settlements_fn is None:
            return jsonify(success=False, error="Market settlements not available"), 501
        try:
            cursor = request.args.get("cursor")
            limit = int(request.args.get("limit", 100))
            result = settlements_fn(cursor=cursor, limit=limit)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/market/bins", methods=["GET"])
    def api_market_bins(league_id=None):
        bins_fn = g.services.market.bins_getter
        if bins_fn is None:
            return jsonify(success=False, error="Market bins not available"), 501
        try:
            result = bins_fn(request.args, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/portfolio/game-positions", methods=["GET"])
    def api_portfolio_game_positions(league_id=None):
        """Get portfolio positions, orders, and fills matched to games on a date."""
        from datetime import timezone as _tz

        date_str = request.args.get("date")
        if not date_str:
            return jsonify(success=False, error="Missing date parameter"), 400
        try:
            game_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify(success=False, error="Invalid date format"), 400

        # Check Kalshi credentials
        api_key = os.environ.get("KALSHI_API_KEY")
        pk_dir = os.environ.get("KALSHI_PRIVATE_KEY_DIR")
        if not api_key or not pk_dir:
            return jsonify(success=True, available=False,
                           message="Kalshi API credentials not configured")

        # Create authenticated connector
        try:
            from sportscore.market import MarketConnector
            connector = MarketConnector({"KALSHI_API_KEY": api_key,
                                         "KALSHI_PRIVATE_KEY_DIR": pk_dir})
        except Exception:
            return jsonify(success=True, available=False,
                           message="Failed to connect to Kalshi API")

        # Get games for date
        league = g.league
        games_coll = league.collections.get("games", "games")
        db = g.db
        query = {"date": date_str}
        # Only filter by league field if the collection is shared across
        # leagues (e.g. soccer's "soccer_games").  Sport apps that use
        # per-league collections (e.g. basketball's "cbb_stats_games")
        # don't store a league field on game documents.
        lid = getattr(league, "league_id", None)
        if lid and lid not in games_coll:
            query["league"] = lid
        game_list = list(db[games_coll].find(
            query,
            {"game_id": 1, "homeTeam.name": 1, "awayTeam.name": 1},
        ))
        if not game_list:
            return jsonify(success=True, available=True, positions={},
                           message="No games for this date")

        # Market config
        mc = (league.raw or {}).get("market", {})
        series_ticker = mc.get("series_ticker", "")
        spread_series = mc.get("spread_series_ticker", "")

        if not series_ticker:
            return jsonify(success=True, available=False,
                           message="No market series_ticker configured")

        # Fetch portfolio data with short-lived cache
        from sportscore.market import SimpleCache
        _cache = SimpleCache(default_ttl=60)
        _TTL = 60

        def _fetch(key, method, **kwargs):
            val = _cache.get(key)
            if val is None:
                try:
                    val = method(**kwargs)
                except Exception:
                    val = []
                _cache.set(key, val, ttl=_TTL)
            return val if isinstance(val, list) else val

        all_positions = _fetch(
            "gp_pos", lambda: connector.get_positions(limit=200).get("market_positions", []))
        min_ts = int((datetime.now(_tz.utc).timestamp() - 86400) * 1000)
        all_fills = _fetch(
            "gp_fills", lambda: connector.get_fills(min_ts=min_ts, limit=200).get("fills", []))
        all_orders = _fetch(
            "gp_orders", lambda: connector.get_orders(status="resting", limit=200).get("orders", []))
        sett_ts = int((datetime.combine(game_date, datetime.min.time()).timestamp() - 86400) * 1000)
        all_settlements = _fetch(
            f"gp_sett:{date_str}",
            lambda: connector.get_settlements(min_ts=sett_ts, limit=200).get("settlements", []))

        # Filter to this sport's tickers
        def matches(item):
            t = item.get("ticker", "") or item.get("event_ticker", "")
            if series_ticker and t.startswith(series_ticker):
                return True
            if spread_series and t.startswith(spread_series):
                return True
            if "MULTIGAME" in t.upper():
                return True
            return False

        all_positions = [p for p in all_positions if matches(p)]
        all_fills = [f for f in all_fills if matches(f)]
        all_orders = [o for o in all_orders if matches(o)]
        all_settlements = [s for s in all_settlements if matches(s)]

        # Match to games
        from sportscore.market.game_markets import match_portfolio_to_games
        result = match_portfolio_to_games(
            games=game_list,
            positions=all_positions,
            fills=all_fills,
            orders=all_orders,
            settlements=all_settlements,
            game_date=game_date,
            league=league,
        )

        return jsonify(
            success=True,
            available=True,
            date=date_str,
            positions=result.game_data,
            fetched_at=datetime.now(_tz.utc).isoformat(),
        )

    # ==================================================================
    # Elo / Cache API (generic — uses sportscore CLI under the hood)
    # ==================================================================

    @app.route("/<league_id>/api/elo/stats", methods=["GET"])
    def api_elo_stats(league_id=None):
        elo_fn = g.services.elo.stats_getter
        if elo_fn is None:
            return jsonify(success=False, error="Elo not available"), 501
        try:
            return jsonify(elo_fn())
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/elo/run", methods=["POST"])
    def api_elo_run(league_id=None):
        elo_runner = g.services.elo.runner
        if elo_runner is None:
            return jsonify(success=False, error="Elo not available"), 501
        try:
            data = request.json or {}
            result = elo_runner(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/elo/clear", methods=["POST"])
    def api_elo_clear(league_id=None):
        elo_clearer = g.services.elo.clearer
        if elo_clearer is None:
            return jsonify(success=False, error="Elo not available"), 501
        try:
            result = elo_clearer(league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/cached-league-stats/stats", methods=["GET"])
    def api_cached_league_stats(league_id=None):
        stats_fn = g.services.elo.cached_league_stats_getter
        if stats_fn is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            return jsonify(stats_fn())
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/cached-league-stats/cache", methods=["POST"])
    def api_cached_league_stats_cache(league_id=None):
        cacher = g.services.elo.league_stats_cacher
        if cacher is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            result = cacher(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/espn-db-audit/audit", methods=["POST"])
    def api_espn_db_audit(league_id=None):
        auditor = g.services.data.espn_db_auditor
        if auditor is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            result = auditor(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500

    @app.route("/<league_id>/api/espn-db-audit/pull", methods=["POST"])
    def api_espn_db_pull(league_id=None):
        puller = g.services.data.espn_db_puller
        if puller is None:
            return jsonify(success=False, error="Not available"), 501
        try:
            data = request.json or {}
            result = puller(data, league=g.league)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify(success=False, error=str(e)), 500
